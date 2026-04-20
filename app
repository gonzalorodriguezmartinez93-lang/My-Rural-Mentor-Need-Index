import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
import base64
from io import BytesIO

st.set_page_config(page_title="My Rural Mentor - Need Index", page_icon="🌱", layout="wide")

# --- 1. CONFIGURACIÓN Y LÓGICA DE DICCIONARIOS ---
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
    if 'si' in v or 'yes' in v or 'sim' in v: return 1
    # Si han escrito un texto relatando un episodio y no está en la lista de negaciones, asume 1
    if len(v) > 4: return 1 
    return 0

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
        
        # Identificadores demográficos
        if ('id' == c_norm or 'nome' in c_norm or 'nombre' in c_norm or 'correo' in c_norm) and not cols['id']: cols['id'] = c
        elif 'xenero' in c_norm or 'genero' in c_norm or 'sexo' in c_norm: cols['gender'] = c
        elif 'orientacion' in c_norm or 'sexual' in c_norm: cols['orientation'] = c
        elif 'etnico' in c_norm or 'etnia' in c_norm: cols['ethnicity'] = c
        
        # Discriminación (busca la palabra en el título de la columna)
        elif 'discrimin' in c_norm:
            cols['discrim'].append(c)
            
        # PWI (Bienestar) - Numérico 0-10 y que NO sea el sentido de comunidad
        elif 'satisfeit' in c_norm or 'satisfech' in c_norm:
            if 'comunidad' not in c_norm: # Se excluye por instrucción explícita
                cols['pwi'].append(c)
                
        # Duke (Apoyo Social) - Escala Likert
        else:
            sample = df[c].dropna().astype(str).str.lower()
            if len(sample) > 0:
                # Comprobar si al menos el 40% de las respuestas coinciden con el texto de la escala Likert
                es_likert = sample.apply(lambda x: any(normalizar(k) in normalizar(x) for k in LIKERT_MAP.keys())).mean()
                if es_likert > 0.4:
                    cols['duke'].append(c)
                    
    return cols

def convertir_likert(val):
    v = normalizar(val)
    for k, num in LIKERT_MAP.items():
        if normalizar(k) in v:
            return num
    return 3 # Neutral por defecto si hay error

# --- 3. MOTOR MATEMÁTICO (Need Index) ---
def calcular_indice(df, cols):
    df_calc = df.copy()
    
    # 1. Duke (Apoyo Social)
    for c in cols['duke']:
        df_calc[c + '_num'] = df_calc[c].apply(convertir_likert)
    df_calc['Support_Raw'] = df_calc[[c + '_num' for c in cols['duke']]].sum(axis=1)
    
    # 2. PWI (Bienestar)
    for c in cols['pwi']:
        df_calc[c] = pd.to_numeric(df_calc[c], errors='coerce').fillna(5) # 5 por defecto si está vacío
    df_calc['WB_Raw'] = df_calc[cols['pwi']].sum(axis=1)
    
    # 3. Discriminación
    if cols['discrim']:
        for c in cols['discrim']:
            df_calc[c + '_bin'] = df_calc[c].apply(es_discriminacion)
        df_calc['Disc_Raw'] = df_calc[[c + '_bin' for c in cols['discrim']]].sum(axis=1)
    else:
        df_calc['Disc_Raw'] = 0
        
    # 4. Vulnerabilidad (Normatividad)
    def calc_norm(row):
        sex = normalizar(row.get(cols['orientation'], ""))
        eth = normalizar(row.get(cols['ethnicity'], ""))
        
        is_lgtb = 0 if 'hetero' in sex or sex == "" else 1
        is_rac = 0 if any(k in eth for k in ETNIAS_NORMATIVAS) or eth == "" else 1
        
        return max(is_lgtb, is_rac)
        
    df_calc['Normativity'] = df_calc.apply(calc_norm, axis=1)
    
    # 5. Cálculo de Z-Scores de la muestra actual (std ddof=0 para población exacta cargada)
    def z_score(series):
        std = series.std(ddof=0)
        return (series - series.mean()) / std if std > 0 else series * 0
        
    z_supp = z_score(df_calc['Support_Raw'])
    z_wb = z_score(df_calc['WB_Raw'])
    z_disc = z_score(df_calc['Disc_Raw'])
    z_norm = z_score(df_calc['Normativity'])
    
    # Invertir factores protectores
    z_supp_inv = -z_supp
    z_wb_inv = -z_wb
    
    # 6. Suma Ponderada
    need_raw = (0.40 * z_supp_inv) + (0.30 * z_wb_inv) + (0.15 * z_disc) + (0.15 * z_norm)
    
    # 7. Indexación Base 100 y Categorización
    df_calc['Need_Index'] = np.clip(50 + (need_raw * 10), 0, 100).round(1)
    df_calc['Prioridad'] = df_calc['Need_Index'].apply(clasificar_prioridad)
    
    return df_calc.sort_values('Need_Index', ascending=False).reset_index(drop=True)

# --- 4. INTERFAZ Y ESTILOS ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-val { font-size: 24px; font-weight: bold; color: #004a99; margin: 0; }
    .metric-title { font-size: 12px; text-transform: uppercase; color: #666; margin: 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Need Index | My Rural Mentor")
st.write("Algoritmo de triaje para calcular la necesidad de intervención sociosanitaria.")

archivo = st.file_uploader("Sube el Excel o CSV de respuestas (Google Forms / Typeform)", type=['xlsx', 'csv'])

if archivo:
    if archivo.name.endswith('.csv'):
        df_raw = pd.read_csv(archivo, sep=None, engine='python')
    else:
        df_raw = pd.read_excel(archivo)
        
    cols_detectadas = autodetectar_columnas(df_raw)
    
    with st.expander("🛠️ Ver Diagnóstico del Detector de Columnas"):
        st.write(f"**Identificador:** {cols_detectadas['id']}")
        st.write(f"**Género:** {cols_detectadas['gender']}")
        st.write(f"**Apoyo Social (Duke):** {len(cols_detectadas['duke'])} preguntas")
        st.write(f"**Bienestar (PWI-7):** {len(cols_detectadas['pwi'])} preguntas")
        st.write(f"**Discriminación:** {len(cols_detectadas['discrim'])} preguntas")

    if st.button("🚀 CALCULAR NEED INDEX"):
        if not cols_detectadas['id']:
            st.error("No se ha encontrado una columna de identificación (Nombre, ID, Correo).")
            st.stop()
            
        with st.spinner('Calculando modelo Z-Score...'):
            df_res = calcular_indice(df_raw, cols_detectadas)
            
            # Asignar Ranking
            df_res.insert(0, 'Ranking', range(1, len(df_res) + 1))
            
            st.success("¡Análisis completado con éxito!")
            
            # --- KPIs ---
            c1, c2, c3, c4 = st.columns(4)
            n_altos = len(df_res[df_res['Need_Index'] >= 60])
            pct_altos = (n_altos / len(df_res)) * 100
            
            with c1: st.markdown(f'<div class="metric-card"><p class="metric-title">Jóvenes Evaluados</p><p class="metric-val">{len(df_res)}</p></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><p class="metric-title">Need Index Medio</p><p class="metric-val">{df_res["Need_Index"].mean():.1f}</p></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><p class="metric-title">Casos Críticos (≥70)</p><p class="metric-val" style="color:#E74C3C">{len(df_res[df_res["Need_Index"] >= 70])}</p></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-card"><p class="metric-title">% Alta Necesidad</p><p class="metric-val">{pct_altos:.1f}%</p></div>', unsafe_allow_html=True)

            st.write("---")
            
            # --- VISUALIZACIONES ---
            t1, t2, t3 = st.tabs(["📋 Tabla de Triaje", "📊 Distribución del Riesgo", "👥 Análisis Demográfico"])
            
            with t1:
                st.write("### Top Jóvenes Prioritarios")
                columnas_mostrar = ['Ranking', cols_detectadas['id'], 'Need_Index', 'Prioridad', 'Normativity', 'Disc_Raw']
                if cols_detectadas['gender']: columnas_mostrar.insert(2, cols_detectadas['gender'])
                
                st.dataframe(df_res[columnas_mostrar], use_container_width=True, hide_index=True)
                
                # Descarga a Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_res[columnas_mostrar].to_excel(writer, index=False, sheet_name='Need_Index')
                st.download_button(label="📥 Descargar Reporte en Excel", data=output.getvalue(), file_name="My_Rural_Mentor_Triage.xlsx", mime="application/vnd.ms-excel")

            with t2:
                fig_hist = px.histogram(df_res, x="Need_Index", nbins=15, color="Prioridad", 
                                        color_discrete_map={'🔴 Muy Alta':'#E74C3C', '🟠 Alta':'#F39C12', '🟡 Media':'#F1C40F', '🟢 Baja':'#27AE60', '🔵 Muy Baja':'#2980B9'},
                                        title="Distribución del Índice de Necesidad (Campana Z)")
                fig_hist.add_vline(x=50, line_dash="dash", line_color="black", annotation_text="Media (50)")
                st.plotly_chart(fig_hist, use_container_width=True)

            with t3:
                if cols_detectadas['gender']:
                    fig_box = px.box(df_res, x=cols_detectadas['gender'], y="Need_Index", color=cols_detectadas['gender'], title="Need Index por Género")
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("No se detectó la columna de género para realizar este cruce.")
