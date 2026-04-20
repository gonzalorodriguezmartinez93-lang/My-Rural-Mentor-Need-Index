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

# --- 2. DETECTOR AUTOMÁTICO INTELIGENTE ---
def autodetectar_columnas(df):
    cols = {'id': None, 'gender': None, 'orientation': None, 'ethnicity': None, 
            'duke': [], 'pwi': [], 'discrim': [], 'community': None}
    
    for c in df.columns:
        c_norm = normalizar(c)
        # Identificadores y Demografía
        if any(x in c_norm for x in ['id', 'nome', 'nombre']) and not cols['id']: cols['id'] = c
        elif any(x in c_norm for x in ['xenero', 'genero']): cols['gender'] = c
        elif 'orientacion' in c_norm or 'sexual' in c_norm: cols['orientation'] = c
        elif 'etnico' in c_norm or 'etnia' in c_norm: cols['ethnicity'] = c
        
        # Especial: Sentido de comunidad (Informativa)
        elif 'sentido da comunidade' in c_norm or 'lugar na sociedade' in c_norm:
            cols['community'] = c
            
        # Discriminación
        elif 'discrimin' in c_norm:
            cols['discrim'].append(c)
            
        # PWI (Bienestar) - Buscamos los que tienen números 0-10
        elif 'satisfeit' in c_norm or 'satisfech' in c_norm:
            if c != cols['community']: cols['pwi'].append(c)
                
        # Duke (Apoyo Social)
        else:
            sample = df[c].dropna().astype(str).str.lower()
            if len(sample) > 0:
                if sample.apply(lambda x: any(normalizar(k) in normalizar(x) for k in LIKERT_MAP.keys())).mean() > 0.4:
                    cols['duke'].append(c)
    return cols

# --- 3. MOTOR DE CÁLCULO ---
def calcular_modelo_tecnico(df, cols):
    df_c = df.copy()
    
    # A. Procesar Apoyo Social (Duke)
    for c in cols['duke']:
        df_c[c + '_n'] = df_c[c].apply(lambda x: next((v for k, v in LIKERT_MAP.items() if normalizar(k) in normalizar(x)), 3))
    df_c['Social_Support'] = df_c[[c + '_n' for c in cols['duke']]].sum(axis=1)
    
    # B. Procesar Bienestar (PWI)
    for c in cols['pwi']:
        df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(5)
    df_c['Well_Being'] = df_c[cols['pwi']].sum(axis=1)
    
    # C. Procesar Discriminación (Detalle de 5 preguntas)
    for c in cols['discrim']:
        df_c[c + '_bin'] = df_c[c].apply(puntuar_discrim)
    df_c['Discrim_Count'] = df_c[[c + '_bin' for c in cols['discrim']]].sum(axis=1)
    
    # D. Recodificar Normatividad
    def recode_norm(row):
        sex = normalizar(row.get(cols['orientation'], ""))
        eth = normalizar(row.get(cols['ethnicity'], ""))
        is_lgtb = "LGTB+" if ('hetero' not in sex and sex != "") else "Norm."
        is_rac = "Rac." if not any(k in eth for k in ETNIAS_NORMATIVAS) and eth != "" else "Norm."
        val_bin = 1 if (is_lgtb == "LGTB+" or is_rac == "Rac.") else 0
        return pd.Series([is_lgtb, is_rac, val_bin])

    df_c[['Sexual_O', 'Ethnicity_Status', 'Norm_Binary']] = df_c.apply(recode_norm, axis=1)
    
    # E. ESTANDARIZACIÓN Z (Muestra actual)
    def z_score(series):
        return (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) > 0 else series * 0
        
    z_supp_inv = -z_score(df_c['Social_Support'])
    z_wb_inv = -z_score(df_c['Well_Being'])
    z_disc = z_score(df_c['Discrim_Count'])
    z_norm = z_score(df_c['Norm_Binary'])
    
    # Peso: 40-30-15-15
    need_raw = (0.40 * z_supp_inv) + (0.30 * z_wb_inv) + (0.15 * z_disc) + (0.15 * z_norm)
    df_c['Need_Index'] = np.clip(50 + (need_raw * 10), 0, 100).round(2)
    
    return df_c.sort_values('Need_Index', ascending=False)

# --- 4. INTERFAZ ---
st.title("🌱 My Rural Mentor - Análisis de Necesidad")
archivo = st.file_uploader("Subir base de datos (Excel/CSV)", type=['xlsx', 'csv'])

if archivo:
    df_in = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df_in.columns = df_in.columns.astype(str).str.strip()
    
    cols = autodetectar_columnas(df_in)
    df_res = calcular_modelo_tecnico(df_in, cols)
    
    # KPIs Estilo Informe
    st.write("### Resumen de la Muestra")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N (Muestra)", len(df_res))
    c2.metric("Media Necesidad", f"{df_res['Need_Index'].mean():.2f}")
    c3.metric("Racializados", len(df_res[df_res['Ethnicity_Status'] == 'Rac.']))
    c4.metric("LGTB+", len(df_res[df_res['Sexual_O'] == 'LGTB+']))

    st.write("---")
    
    # TABLA TÉCNICA (List of potential participants)
    st.write("### 📋 Listado de Priorización de Participantes")
    
    # Preparar tabla limpia para visualización
    tabla_tec = df_res.copy()
    tabla_tec = tabla_tec[[cols['id'], cols['gender'], 'Sexual_O', 'Ethnicity_Status', 
                           'Social_Support', 'Well_Being', 'Discrim_Count', 
                           cols['community'], 'Need_Index']]
    
    # Renombrar para que sea idéntico a tu informe
    tabla_tec.columns = ['ID', 'Gender', 'Sexual O.', 'Ethnicity', 'Social Support', 
                         'Well-Being', 'Discr.', 'Sense of Community', 'Indicator of Need']

    # Estilo condicional para detectar casos críticos
    def style_critical(v):
        if isinstance(v, float) and v >= 70: return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        if isinstance(v, float) and v >= 60: return 'background-color: #fff3cd; color: #856404'
        return ''

    st.dataframe(tabla_tec.style.applymap(style_critical, subset=['Indicator of Need']), use_container_width=True, hide_index=True)

    # Gráfico de distribución
    st.write("### 📊 Perfil de Vulnerabilidad")
    fig = px.scatter(df_res, x="Social_Support", y="Well_Being", size="Need_Index", color="Ethnicity_Status",
                     hover_name=cols['id'], title="Relación Apoyo-Bienestar y Necesidad (Tamaño del círculo)")
    st.plotly_chart(fig, use_container_width=True)

    # Exportación
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        tabla_tec.to_excel(writer, index=False, sheet_name='Triaje')
    st.download_button("📥 Descargar Tabla de Priorización (Excel)", output.getvalue(), "MRM_Priorizacion_Tecnica.xlsx")
