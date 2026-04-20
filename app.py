import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from io import BytesIO

st.set_page_config(page_title="Need Index | My Rural Mentor", page_icon="🌱", layout="wide")

# --- 1. FUNCIONES DE NORMALIZACIÓN ---
def normalizar(s):
    if pd.isna(s): return ""
    s = str(s).lower().replace('\n', ' ').strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def zscore(x):
    # Usamos ddof=0 tal como indica la instrucción exacta
    std = x.std(ddof=0)
    if std == 0: return x * 0
    return (x - x.mean()) / std

# --- 2. DETECTOR DE COLUMNAS ---
def detectar_columnas_mentor(df):
    cols = {'id': None, 'gender': None, 'orientation': None, 'ethnicity': None, 
            'duke': [], 'pwi': [], 'discrim': [], 'community': None}
    
    for c in df.columns:
        c_norm = normalizar(c)
        # Demografía
        if any(x in c_norm for x in ['id', 'nome', 'nombre']) and not cols['id']: cols['id'] = c
        elif any(x in c_norm for x in ['xenero', 'genero']): cols['gender'] = c
        elif 'orientacion' in c_norm or 'sexual' in c_norm: cols['orientation'] = c
        elif 'etnico' in c_norm or 'etnia' in c_norm: cols['ethnicity'] = c
        # Especiales
        elif 'comunidade' in c_norm or 'sociedade' in c_norm: cols['community'] = c
        elif 'discrimin' in c_norm: cols['discrim'].append(c)
        # PWI (0-10) y Duke (Texto)
        elif 'satisfeit' in c_norm or 'satisfech' in c_norm:
            if c != cols['community']: cols['pwi'].append(c)
        else:
            sample = df[c].dropna().astype(str).str.lower()
            if len(sample) > 0 and sample.str.contains('tanto como quero|nin moito').any():
                cols['duke'].append(c)
    return cols

# --- 3. PROCESAMIENTO SEGÚN INSTRUCCIONES EXACTAS ---
def procesar_need_index(df, cols):
    df_c = df.copy()
    
    # A. Cuantificar Duke (1-5)
    likert = {'moito menos': 1, 'menos do que': 2, 'nin moito': 3, 'case tanto': 4, 'tanto como': 5}
    for c in cols['duke']:
        df_c[c+'_n'] = df_c[c].apply(lambda x: next((v for k, v in likert.items() if k in normalizar(x)), 3))
    df_c['Social Support'] = df_c[[c+'_n' for c in cols['duke']]].sum(axis=1)

    # B. Cuantificar PWI (0-10)
    for c in cols['pwi']:
        df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(5)
    df_c['Well-Being'] = df_c[cols['pwi']].sum(axis=1)

    # C. Cuantificar Discriminación (Suma de los 5 ítems)
    def es_si(val):
        v = normalizar(val)
        return 1 if any(x in v for x in ['si', 'sim', 'verdade']) else 0
    
    for c in cols['discrim']:
        df_c[c+'_b'] = df_c[c].apply(es_si)
    df_c['Discrimination'] = df_c[[c+'_b' for c in cols['discrim']]].sum(axis=1)

    # D. Crear Variable NORMATIVITY (OR lógico)
    def calc_normativity(row):
        sex = normalizar(row.get(cols['orientation'], ""))
        eth = normalizar(row.get(cols['ethnicity'], ""))
        # Es 1 si NO es hetero O si es racializado
        es_lgtb = 1 if ('hetero' not in sex and sex != "") else 0
        es_rac = 1 if not any(k in eth for k in ['espanol', 'galego', 'blanco']) and eth != "" else 0
        return 1 if (es_lgtb == 1 or es_rac == 1) else 0

    df_c['Normativity'] = df_c.apply(calc_normativity, axis=1)

    # E. CALCULO ESTADÍSTICO (PASOS 2 A 6 DE LA LISTA)
    df_c['ZSupport'] = zscore(df_c['Social Support'])
    df_c['ZWell'] = zscore(df_c['Well-Being'])
    df_c['ZDisc'] = zscore(df_c['Discrimination'])
    df_c['ZNorm'] = zscore(df_c['Normativity'])

    # Ponderación con inversión previa (-Z) para Support y Well
    df_c['Need_raw'] = (
        0.40 * (-df_c['ZSupport']) +
        0.30 * (-df_c['ZWell']) +
        0.15 * df_c['ZDisc'] +
        0.15 * df_c['ZNorm']
    )

    # Transformación final
    df_c['Indicator of Need'] = (50 + (df_c['Need_raw'] * 10)).clip(0, 100).round(2)
    
    return df_c.sort_values('Indicator of Need', ascending=False)

# --- 4. INTERFAZ ---
st.title("🌱 Need Index | My Rural Mentor")
archivo = st.file_uploader("Subir base de datos", type=['xlsx', 'csv'])

if archivo:
    df_in = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df_in.columns = df_in.columns.astype(str).str.strip()
    
    cols = detectar_columnas_mentor(df_in)
    df_res = procesar_need_index(df_in, cols)
    
    st.write("### 📋 Resultados del Triaje")
    
    # Tabla final formateada
    def label_sex(val):
        sex = normalizar(val)
        return 'LGTB+' if ('hetero' not in sex and sex != "") else 'Norm.'
    
    def label_eth(val):
        eth = normalizar(val)
        return 'Rac.' if not any(k in eth for k in ['espanol', 'galego', 'blanco']) and eth != "" else 'Norm.'

    df_res['Sexual O.'] = df_res[cols['orientation']].apply(label_sex)
    df_res['Ethnicity'] = df_res[cols['ethnicity']].apply(label_eth)
    
    # Formatear Gender a M/W
    def label_gender(val):
        v = normalizar(val)
        if any(x in v for x in ['home', 'hombre', 'man']): return 'M'
        if any(x in v for x in ['muller', 'mujer', 'woman']): return 'W'
        return 'O'
    df_res['Gender_Abr'] = df_res[cols['gender']].apply(label_gender)

    tabla_final = df_res[[cols['id'], 'Gender_Abr', 'Sexual O.', 'Ethnicity', 
                          'Social Support', 'Well-Being', 'Discrimination', 
                          cols['community'], 'Indicator of Need']]
    
    tabla_final.columns = ['ID', 'Gender', 'Sexual O.', 'Ethnicity', 'Social Support', 
                          'Well-Being', 'Discr.', 'Sense of Community', 'Indicator of Need']

    st.dataframe(tabla_final, use_container_width=True, hide_index=True)

    # Botón Descarga
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tabla_final.to_excel(writer, index=False, sheet_name='Need_Index')
    st.download_button("📥 Descargar Excel", output.getvalue(), "MRM_Results.xlsx")
