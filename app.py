import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from io import BytesIO

st.set_page_config(page_title="Need Index | Auditoría Técnica", page_icon="🌱", layout="wide")

# --- FUNCIONES DE LIMPIEZA ---
def normalizar(s):
    if pd.isna(s): return ""
    s = str(s).lower().replace('\n', ' ').strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def transformar_genero(val):
    v = normalizar(val)
    if any(x in v for x in ['home', 'hombre', 'man', 'macho']): return 'M'
    if any(x in v for x in ['muller', 'mujer', 'woman', 'femia']): return 'W'
    return 'O'

def puntuar_si_no(val):
    v = normalizar(val)
    # Si la respuesta contiene 'si', 'si', 'verdadeiro' o no es una negación clara
    if any(x in v for x in ['si', 'si', 'sim', 'verdade']): return 1
    return 0

# --- MOTOR DE CÁLCULO REVISADO ---
def calcular_indice_exacto(df):
    df_c = df.copy()
    
    # 1. Identificar Columnas por posición/contenido para evitar fallos de nombre
    # Duke: 11 ítems (Escala Likert Texto)
    # PWI: 7 ítems (0-10) -> El 6º suele ser Comunidad
    # Discrim: 5 ítems (Si/No)
    
    duke_cols = []
    pwi_cols = []
    discrim_cols = []
    community_col = None
    
    for c in df.columns:
        c_norm = normalizar(c)
        sample = df[c].dropna().astype(str).unique()
        
        if 'discrimin' in c_norm:
            discrim_cols.append(c)
        elif 'satisfeit' in c_norm or 'satisfech' in c_norm:
            if 'comunidade' in c_norm or 'sociedade' in c_norm or 'lugar na' in c_norm:
                community_col = c
            else:
                pwi_cols.append(c)
        elif any(k in str(sample).lower() for k in ['tanto como quero', 'nin moito']):
            duke_cols.append(c)

    # --- CÁLCULOS NUMÉRICOS ---
    
    # APOYO SOCIAL (Duke - 11 ítems)
    likert_map = {'moito menos': 1, 'menos do que': 2, 'nin moito': 3, 'case tanto': 4, 'tanto como': 5}
    for c in duke_cols:
        df_c[c+'_n'] = df_c[c].apply(lambda x: next((v for k, v in likert_map.items() if k in normalizar(x)), 3))
    df_c['Social_Support'] = df_c[[c+'_n' for c in duke_cols]].sum(axis=1)

    # BIENESTAR (PWI - 6 ítems, excluyendo comunidad)
    for c in pwi_cols:
        df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(df[c].mean() if df[c].mean() else 5)
    df_c['Well_Being'] = df_c[pwi_cols].sum(axis=1)

    # DISCRIMINACIÓN (Suma de los 5 ítems)
    for c in discrim_cols:
        df_c[c+'_b'] = df_c[c].apply(puntuar_si_no)
    df_c['Discr_Count'] = df_c[[c+'_b' for c in discrim_cols]].sum(axis=1)

    # NORMATIVIDAD
    # (Buscamos columnas de orientación y etnia)
    orient_col = next((c for c in df.columns if 'orientacion' in normalizar(c)), None)
    eth_col = next((c for c in df.columns if 'etnia' in normalizar(c) or 'etnico' in normalizar(c)), None)
    
    def check_norm(row):
        sex = normalizar(row.get(orient_col, ""))
        eth = normalizar(row.get(eth_col, ""))
        is_lgtb = 1 if ('hetero' not in sex and sex != "") else 0
        is_rac = 1 if not any(k in eth for k in ['espanol', 'galego', 'blanco']) and eth != "" else 0
        return max(is_lgtb, is_rac)
    
    df_c['Norm_Binary'] = df_c.apply(check_norm, axis=1)

    # --- PONDERACIÓN Z-SCORE (std ddof=0) ---
    def z(series):
        std = series.std(ddof=0)
        return (series - series.mean()) / std if std > 0 else series * 0

    # Need_raw = 40% Supp(inv) + 30% WB(inv) + 15% Disc + 15% Norm
    raw = (0.40 * -z(df_c['Social_Support'])) + \
          (0.30 * -z(df_c['Well_Being'])) + \
          (0.15 * z(df_c['Discr_Count'])) + \
          (0.15 * z(df_c['Norm_Binary']))
    
    df_c['Indicator of Need'] = (50 + (raw * 10)).clip(0, 100).round(2)
    
    return df_c, duke_cols, pwi_cols, discrim_cols, community_col

# --- INTERFAZ ---
st.title("🌱 Auditoría de Need Index")
archivo = st.file_uploader("Sube el archivo", type=['xlsx', 'csv'])

if archivo:
    df = pd.read_csv(archivo, sep=None, engine='python') if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df_res, d_cols, p_cols, dis_cols, com_col = calcular_indice_exacto(df)
    
    st.info(f"Auditoría de columnas: Duke ({len(d_cols)}), PWI ({len(p_cols)}), Discrim ({len(dis_cols)})")
    
    # Tabla de resultados
    id_col = next((c for c in df.columns if 'id' in normalizar(c) or 'nome' in normalizar(c)), df.columns[0])
    gen_col = next((c for c in df.columns if 'xenero' in normalizar(c)), None)
    
    final_table = df_res[[id_col, 'Social_Support', 'Well_Being', 'Discr_Count', 'Indicator of Need']].copy()
    st.dataframe(final_table.sort_values('Indicator of Need', ascending=False), use_container_width=True)
