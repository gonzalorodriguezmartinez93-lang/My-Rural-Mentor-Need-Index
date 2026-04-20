import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Need Index | My Rural Mentor", page_icon="🌱", layout="wide")

# --- 1. DICCIONARIOS Y NORMALIZACIÓN ---
LIKERT_MAP = {
    'moito menos do que quero': 1, 'menos do que quero': 2,
    'nin moito nin pouco': 3, 'case tanto como quero': 4, 'tanto como quero': 5
}

ETNIAS_NORMATIVAS = ['espanol', 'galego', 'blanco', 'caucasico', 'europeo', 'español', 'gallego']

def normalizar(s):
    if pd.isna(s): return ""
    s = str(s).lower().replace('\n', ' ').strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def puntuar_discrim(val):
    v = normalizar(val)
    if not v or v in ['non', 'no', 'nada', 'nunca', 'ningunha', 'ninguna']: return 0
    return 1

# --- 2. DETECTOR AUTOMÁTICO ---
def autodetectar_columnas(df):
    cols = {'id': None, 'gender': None, 'orientation': None, 'ethnicity': None, 
            'duke': [], 'pwi': [], 'discrim': [], 'community': None}
    
    for c in df.columns:
        c_norm = normalizar(c)
        if any(x in c_norm for x in ['id', 'nome', 'nombre']) and not cols['id']: cols['id'] = c
        elif any(x in c_norm for x in ['xenero', 'genero']): cols['gender'] = c
        elif 'orientacion' in c_norm or 'sexual' in c_norm: cols['orientation'] = c
        elif 'etnico' in c_norm or 'etnia' in c_norm: cols['ethnicity'] = c
        elif 'sentido da comunidade' in c_norm or 'lugar na sociedade' in c_norm: cols['community'] = c
        elif 'discrimin' in c_norm: cols['discrim'].append(c)
        elif 'satisfeit' in c_norm or 'satisfech' in c_norm:
            if c != cols['community']: cols['pwi'].append(c)
        else:
            sample = df[c].dropna().astype(str).str.lower()
            if len(sample) > 0 and sample.apply(lambda x: any(normalizar(k) in normalizar(x) for k in LIKERT_MAP.keys())).mean() > 0.4:
                cols['duke'].append(c)
    return cols

# --- 3. MOTOR DE CÁLCULO ---
def calcular_modelo_tecnico(df, cols):
    df_c = df.copy()
    
    # Duke y PWI
    for c in cols['duke']:
        df_c[c + '_n'] = df_c[c].apply(lambda x: next((v for k, v in LIKERT_MAP.items() if normalizar(k) in normalizar(x)), 3))
    df_c['Social_Support'] = df_c[[c + '_n' for c in cols['duke']]].sum(axis=1)
    
    for c in cols['pwi']:
        df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(5)
    df_c['Well_Being'] = df_c[cols['pwi']].sum(axis=1)
    
    # Discriminación Detallada
    for i, c in enumerate(cols['discrim']):
        df_c[f'D{i+1}'] = df_c[c].apply(puntuar_discrim)
    df_c['Discr.'] = df_c[[f'D{i+1}' for i in range(len(cols['discrim']))]].sum(axis=1)
    
    # Normatividad
    def recode_norm(row):
        sex = normalizar(row.get(cols['orientation'], ""))
        eth = normalizar(row.get(cols['ethnicity'], ""))
        is_lgtb = "LGTB+" if ('hetero' not in sex and sex != "") else "Norm."
        is_rac = "Rac." if not any(k in eth for k in ETNIAS_NORMATIVAS) and eth != "" else "Norm."
        val_bin = 1 if (is_lgtb == "LGTB+" or is_rac == "Rac.") else 0
        return pd.Series([is_lgtb, is_rac, val_bin])

    df_c[['Sexual O.', 'Ethnicity', 'Norm_Binary']] = df_c.apply(recode_norm, axis=1)
    
    # Z-Scores
    def z_score(series):
        return (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) > 0 else series * 0
        
    z_supp_inv = -z_score(df_c['Social_Support'])
    z_wb_inv = -z_score(df_c['Well_Being'])
    z_disc = z_score(df_c['Discr.'])
    z_norm = z_score(df_c['Norm_Binary'])
    
    need_raw = (0.40 * z_supp_inv) + (0.30 * z_wb_inv) + (0.15 * z_disc) + (0.15 * z_norm)
    df_c['Indicator of Need'] = np.clip(50 + (need_raw * 10), 0, 100).round(2)
    
    return df_c.sort_values('Indicator of Need', ascending=False)

# --- 4. INTERFAZ ---
st.title("🌱 Need Index | My Rural Mentor")
archivo = st.file_uploader("Subir base de datos", type=['xlsx', 'csv'])

if archivo:
    df_in = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df_in.columns = df_in.columns.astype(str).str.strip()
    
    cols = autodetectar_columnas(df_in)
    df_res = calcular_modelo_tecnico(df_in, cols)
    
    st.write("### 📋 Listado de Priorización Técnica")
    
    # Selección de columnas para la tabla final
    tabla_final = df_res[[cols['id'], cols['gender'], 'Sexual O.', 'Ethnicity', 
                          'Social_Support', 'Well_Being', 'Discr.', 
                          cols['community'], 'Indicator of Need']].copy()
    
    # Renombrar para el reporte
    tabla_final.columns = ['ID', 'Gender', 'Sexual O.', 'Ethnicity', 'Social Support', 
                          'Well-Being', 'Discr.', 'Sense of Community', 'Indicator of Need']

    # Función de estilo corregida (usando .map en lugar de .applymap)
    def style_rows(val):
        if isinstance(val, (int, float)):
            if val >= 70: return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            if val >= 60: return 'background-color: #fff3cd; color: #856404;'
        return ''

    st.dataframe(tabla_final.style.map(style_rows, subset=['Indicator of Need']), 
                 use_container_width=True, hide_index=True)

    # Exportar
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tabla_final.to_excel(writer, index=False, sheet_name='Triage')
    st.download_button("📥 Descargar Tabla de Priorización", output.getvalue(), "MRM_Triage_Results.xlsx")
