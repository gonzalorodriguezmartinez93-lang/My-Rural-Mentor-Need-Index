import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
from scipy import stats
from io import BytesIO

st.set_page_config(page_title="Need Index | My Rural Mentor", page_icon="🌱", layout="wide")

# --- 1. CONFIGURACIÓN Y TRADUCCIONES ---
LIKERT_MAP = {
    'moito menos do que quero': 1,
    'menos do que quero': 2,
    'nin moito nin pouco': 3,
    'case tanto como quero': 4,
    'tanto como quero': 5
}

ETNIAS_NORMATIVAS = ['espanol', 'galego', 'blanco', 'caucasico', 'europeo', 'español', 'gallego']

def normalizar(s):
    if pd.isna(s): return ""
    s = str(s).lower().replace('\n', ' ').strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def es_discriminacion(val):
    v = normalizar(val)
    if not v or v in ['non', 'no', 'nada', 'nunca', 'ningunha', 'ninguna']: return 0
    return 1 # Cualquier respuesta positiva o relato de incidente suma 1

def clasificar_prioridad(score):
    if score >= 70: return '🔴 Muy Alta'
    if score >= 60: return '🟠 Alta'
    if score >= 50: return '🟡 Media'
    if score >= 40: return '🟢 Baja'
    return '🔵 Muy Baja'

# --- 2. DETECTOR AUTOMÁTICO DE COLUMNAS ---
def autodetectar_columnas(df):
    cols = {'id': None, 'gender': None, 'orientation': None, 'ethnicity': None, 
            'duke': [], 'pwi': [], 'discrim': []}
    
    for c in df.columns:
        c_norm = normalizar(c)
        
        # Identificadores
        if any(x in c_norm for x in ['id', 'nome', 'nombre', 'apelidos']) and not cols['id']: cols['id'] = c
        elif any(x in c_norm for x in ['xenero', 'genero', 'sexo']): cols['gender'] = c
        elif 'orientacion' in c_norm or 'sexual' in c_norm: cols['orientation'] = c
        elif 'etnico' in c_norm or 'etnia' in c_norm: cols['ethnicity'] = c
        
        # Discriminación
        elif 'discrimin' in c_norm:
            cols['discrim'].append(c)
            
        # PWI (Bienestar) - Escala 0-10
        elif 'satisfeit' in c_norm or 'satisfech' in c_norm:
            if 'comunidade' not in c_norm and 'comunidad' not in c_norm: 
                cols['pwi'].append(c)
                
        # Duke (Apoyo Social) - Escala Likert Texto
        else:
            sample = df[c].dropna().astype(str).str.lower()
            if len(sample) > 0:
                coincidencias = sample.apply(lambda x: any(normalizar(k) in normalizar(x) for k in LIKERT_MAP.keys())).mean()
                if coincidencias > 0.4:
                    cols['duke'].append(c)
                    
    return cols

def convertir_likert(val):
    v = normalizar(val)
    for k, num in LIKERT_MAP.items():
        if normalizar(k) in v: return num
    return 3

# --- 3. MOTOR DE CÁLCULO ---
def calcular_modelo(df, cols):
    df_c = df.copy()
    
    # Puntuaciones Directas
    # Duke: Suma de ítems (11-55)
    for c in cols['duke']:
        df_c[c + '_n'] = df_c[c].apply(convertir_likert)
    df_c['Support_Raw'] = df_c[[c + '_n' for c in cols['duke']]].sum(axis=1)
    
    # PWI: Suma de ítems (0-60, excluyendo comunidad)
    for c in cols['pwi']:
        df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(5)
    df_c['WB_Raw'] = df_c[cols['pwi']].sum(axis=1)
    
    # Discriminación: Conteo de 'Sís'
    if cols['discrim']:
        for c in cols['discrim']:
            df_c[c + '_b'] = df_c[c].apply(es_discriminacion)
        df_c['Disc_Raw'] = df_c[[c + '_b' for c in cols['discrim']]].sum(axis=1)
    else:
        df_c['Disc_Raw'] = 0
        
    # Normatividad (Binaria)
    def check_norm(row):
        sex = normalizar(row.get(cols['orientation'], ""))
        eth = normalizar(row.get(cols['ethnicity'], ""))
        lgtb = 0 if 'hetero' in sex or sex == "" else 1
        rac = 0 if any(k in eth for k in ETNIAS_NORMATIVAS) or eth == "" else 1
        return max(lgtb, rac)
        
    df_c['Normativity'] = df_c.apply(check_norm, axis=1)
    
    # ESTANDARIZACIÓN Z-SCORE
    def z_score(series):
        std = series.std(ddof=0)
        return (series - series.mean()) / std if std > 0 else series * 0
        
    z_supp_inv = -z_score(df_c['Support_Raw'])
    z_wb_inv = -z_score(df_c['WB_Raw'])
    z_disc = z_score(df_c['Disc_Raw'])
    z_norm = z_score(df_c['Normativity'])
    
    # Índice Bruto Ponderado
    need_raw = (0.40 * z_supp_inv) + (0.30 * z_wb_inv) + (0.15 * z_disc) + (0.15 * z_norm)
    
    # Transformación Base 100
    df_c['Need_Index'] = np.clip(50 + (need_raw * 10), 0, 100).round(1)
    df_c['Prioridad'] = df_c['Need_Index'].apply(clasificar_prioridad)
    
    return df_c.sort_values('Need_Index', ascending=False).reset_index(drop=True)

# --- 4. INTERFAZ ---
st.title("🌱 Need Index | My Rural Mentor")
st.markdown("Herramienta profesional de triaje para la detección de vulnerabilidad social.")

archivo = st.file_uploader("Subir respuestas del formulario (.xlsx, .csv)", type=['xlsx', 'csv'])

if archivo:
    df_in = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df_in.columns = df_in.columns.astype(str).str.strip()
    
    cols = autodetectar_columnas(df_in)
    
    with st.sidebar:
        st.header("⚙️ Diagnóstico")
        st.write(f"**Duke (Apoyo):** {len(cols['duke'])} ítems")
        st.write(f"**PWI (Bienestar):** {len(cols['pwi'])} ítems")
        st.write(f"**Discriminación:** {len(cols['discrim'])} ítems")

    if st.button("🚀 CALCULAR ÍNDICE DE NECESIDAD"):
        df_final = calcular_modelo(df_in, cols)
        
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Jóvenes", len(df_final))
        with c2: st.metric("Media Grupo", f"{df_final['Need_Index'].mean():.1f}")
        with c3: st.metric("Casos Críticos", len(df_final[df_final['Need_Index'] >= 70]))
        with c4: st.metric("Alta Necesidad", f"{(len(df_final[df_final['Need_Index'] >= 60])/len(df_final)*100):.1f}%")

        st.divider()
        
        tab1, tab2 = st.tabs(["📋 Ranking de Prioridad", "📊 Análisis Visual"])
        
        with tab1:
            resumen = df_final[[cols['id'], 'Need_Index', 'Prioridad', 'Normativity', 'Disc_Raw']]
            st.dataframe(resumen, use_container_width=True, hide_index=True)
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_final.to_excel(writer, index=False, sheet_name='Resultados')
            st.download_button("📥 Descargar Excel Completo", output.getvalue(), "Prioridades_Rural_Mentor.xlsx")

        with tab2:
            fig = px.histogram(df_final, x="Need_Index", color="Prioridad", 
                               nbins=20, title="Distribución de Necesidad en la Muestra",
                               color_discrete_map={'🔴 Muy Alta':'#E74C3C', '🟠 Alta':'#F39C12', '🟡 Media':'#F1C40F', '🟢 Baja':'#27AE60', '🔵 Muy Baja':'#2980B9'})
            st.plotly_chart(fig, use_container_width=True)
