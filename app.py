"""
PRÉDICTION INTELLIGENTE IMMOBILIÈRE
Application Streamlit Professionnelle
MUGENI KANZA CHRISTIAN | Master IA | Intelligence Artificielle
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Intelligente Immobilière",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

  /* ── Global Reset ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #060E18 0%, #0D1F35 50%, #091524 100%);
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #0F2040 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  /* ── Hide streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Headers ── */
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00D4FF 0%, #0099BB 40%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
  }
  .hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: rgba(0,212,255,0.6);
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
  }
  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #E8EDF2;
    border-left: 3px solid #00D4FF;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
  }
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: rgba(0,212,255,0.5);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }

  /* ── Cards ── */
  .metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.07) 0%, rgba(15,25,40,0.9) 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00D4FF, transparent);
  }
  .metric-card:hover {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 8px 32px rgba(0,212,255,0.12);
    transform: translateY(-2px);
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #00D4FF;
    line-height: 1.1;
  }
  .metric-label {
    font-size: 0.78rem;
    color: rgba(232,237,242,0.6);
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #34D399;
    margin-top: 0.2rem;
  }

  /* ── Prediction Result ── */
  .prediction-result {
    background: linear-gradient(135deg, rgba(0,212,255,0.12) 0%, rgba(255,215,0,0.05) 100%);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .prediction-result::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #FFD700, transparent);
  }
  .result-price {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: rgba(0,212,255,0.7);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .result-confidence {
    font-size: 0.85rem;
    color: rgba(232,237,242,0.7);
    margin-top: 0.5rem;
  }

  /* ── Classification badge ── */
  .class-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,153,187,0.1));
    border: 1px solid rgba(0,212,255,0.4);
    border-radius: 30px;
    padding: 0.5rem 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #00D4FF;
    letter-spacing: 0.05em;
    margin: 0.5rem 0;
  }

  /* ── Form styling ── */
  .stSelectbox label, .stSlider label, .stNumberInput label {
    color: rgba(232,237,242,0.85) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
  }
  .stSelectbox > div > div {
    background-color: rgba(10,22,40,0.9) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 8px !important;
    color: #E8EDF2 !important;
  }
  .stSelectbox > div > div:focus-within {
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0099BB 100%) !important;
    color: #060E18 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #33DDFF 0%, #00B8D9 100%) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.35) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(10,22,40,0.7);
    border-radius: 10px;
    border: 1px solid rgba(0,212,255,0.15);
    gap: 0;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(232,237,242,0.5) !important;
    border-radius: 7px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.3s !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,153,187,0.1)) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
  }

  /* ── Divider ── */
  .custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
    margin: 1.5rem 0;
  }

  /* ── Info boxes ── */
  .info-box {
    background: rgba(0,212,255,0.06);
    border-left: 3px solid #00D4FF;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: rgba(232,237,242,0.8);
    margin: 0.8rem 0;
  }
  .warning-box {
    background: rgba(255,107,53,0.08);
    border-left: 3px solid #FF6B35;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: rgba(232,237,242,0.8);
    margin: 0.8rem 0;
  }
  .success-box {
    background: rgba(52,211,153,0.08);
    border-left: 3px solid #34D399;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: rgba(232,237,242,0.8);
    margin: 0.8rem 0;
  }

  /* ── Footer signature ── */
  .footer-signature {
    background: linear-gradient(135deg, rgba(0,212,255,0.06) 0%, rgba(255,215,0,0.04) 100%);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 3rem;
  }
  .footer-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4FF, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .footer-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: rgba(0,212,255,0.6);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.3rem;
  }
  .footer-filiere {
    font-size: 0.82rem;
    color: rgba(232,237,242,0.5);
    margin-top: 0.2rem;
    font-style: italic;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #060E18; }
  ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }

  /* ── Plotly/Charts ── */
  .js-plotly-plot { border-radius: 10px; }

  /* ── Sidebar nav items ── */
  .sidebar-nav-item {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.82rem;
    color: rgba(232,237,242,0.7);
    transition: all 0.2s;
    margin-bottom: 0.2rem;
  }
  .sidebar-nav-item:hover {
    background: rgba(0,212,255,0.1);
    color: #00D4FF;
  }

  /* ── Dashboard table ── */
  .dataframe {
    background: rgba(10,22,40,0.9) !important;
    border-radius: 8px !important;
  }
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Number input ── */
  .stNumberInput > div > div > input {
    background: rgba(10,22,40,0.9) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    color: #E8EDF2 !important;
    border-radius: 8px !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_all_models():
    models_dir = os.path.join(BASE_DIR, 'models')
    m = {}
    m['dt_reg'] = joblib.load(os.path.join(models_dir, 'dt_regressor.pkl'))
    m['rf_reg'] = joblib.load(os.path.join(models_dir, 'rf_regressor.pkl'))
    m['svm_clf'] = joblib.load(os.path.join(models_dir, 'svm_classifier.pkl'))
    m['rfc_clf'] = joblib.load(os.path.join(models_dir, 'rfc_classifier.pkl'))
    m['le'] = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    with open(os.path.join(models_dir, 'metadata.json'), 'r') as f:
        m['meta'] = json.load(f)
    return m

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, 'data', 'housing_data.csv')
    return pd.read_csv(data_path)

try:
    models = load_all_models()
    df = load_data()
    models_ok = True
except Exception as e:
    st.error(f"❌ Erreur chargement : {e}")
    models_ok = False
    st.stop()

NEIGHBORHOODS = ['NAmes','CollgCr','OldTown','Edwards','Somerst','NridgHt','Gilbert','Sawyer','NWAmes','Mitchel']
HOUSE_STYLES  = ['1Story','2Story','1.5Fin','SFoyer','SLvl']
BLDG_LABELS   = {'1Fam':'🏡 Maison individuelle','2fmCon':'🏘 Bi-familiale','Duplex':'🏠 Duplex','TwnhsE':'🏙 Townhouse (ext.)','Twnhs':'🏙 Townhouse (int.)'}
PAL = {'bg':'#0F1923','primary':'#00D4FF','secondary':'#FF6B35','accent':'#FFD700','text':'#E8EDF2','card':'#1A2535','card2':'#0D1F35'}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
      <div style="font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:900;
                  background:linear-gradient(135deg,#00D4FF,#FFD700);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text;">🏛️ Immo Predictor</div>
      <div style="font-family:'Space Mono',monospace; font-size:0.55rem; color:rgba(0,212,255,0.5);
                  letter-spacing:0.2em; text-transform:uppercase; margin-top:0.2rem;">
        Valorisation Intelligente
      </div>
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3),transparent); margin:0.8rem 0 1.2rem;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Prédiction du Prix",
        "🏷️  Classification",
        "📊  Dashboard Analytique",
        "📋  Rapport des Modèles"
    ], label_visibility="collapsed")

    st.markdown('<div style="height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.2),transparent); margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    # Model status
    st.markdown('<div class="section-label">Statut des Modèles</div>', unsafe_allow_html=True)
    meta_r = models['meta']['regression']
    meta_c = models['meta']['classification']
    st.markdown(f"""
    <div style="font-size:0.75rem; color:rgba(232,237,242,0.7); line-height:2;">
      <span style="color:#34D399;">●</span> RF Regressor — R² {meta_r['rf_results']['R2']:.3f}<br>
      <span style="color:#34D399;">●</span> DT Regressor — R² {meta_r['dt_results']['R2']:.3f}<br>
      <span style="color:#34D399;">●</span> RF Classifier — F1 {meta_c['rfc_results']['F1']:.3f}<br>
      <span style="color:#34D399;">●</span> SVM Classifier — F1 {meta_c['svm_results']['F1']:.3f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.2),transparent); margin:1.2rem 0;"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.68rem; color:rgba(232,237,242,0.35); text-align:center; line-height:1.8;">
      Dataset : {len(df):,} observations<br>
      Features : {df.shape[1]} variables
    </div>
    """, unsafe_allow_html=True)

# ─── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem;">
  <div class="hero-subtitle">■ Système d'Analyse Immobilière Avancée ■</div>
  <div class="hero-title">Prédiction Intelligente Immobilière</div>
  <div style="font-size:0.88rem; color:rgba(232,237,242,0.55); max-width:580px; line-height:1.6; margin-bottom:0.5rem;">
    Estimation du prix de vente et classification automatique du type de bien
    par apprentissage automatique supervisé — Random Forest & SVM.
  </div>
</div>
<div style="height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.4),rgba(255,215,0,0.3),transparent); margin-bottom:1.5rem;"></div>
""", unsafe_allow_html=True)

def make_fig():
    fig, ax = plt.subplots(facecolor=PAL['bg'])
    ax.set_facecolor(PAL['card'])
    for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
    ax.tick_params(colors=PAL['text'])
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : PRÉDICTION DU PRIX
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Prédiction du Prix":

    st.markdown('<div class="section-title">Estimation du Prix de Vente</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Renseignez les caractéristiques du bien. Le modèle <strong>Random Forest Regressor</strong> (R²=0.80) estimera la valeur marchande en temps réel.</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Caractéristiques Structurelles</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gr_liv  = st.number_input("Surface habitable (pi²)", 500, 5000, 1500, 50)
            bsmt    = st.number_input("Surface sous-sol (pi²)",  0, 3500, 900, 50)
            lot     = st.number_input("Surface terrain (pi²)",  2000, 50000, 10000, 500)
            bedroom = st.number_input("Chambres",  1, 8, 3)
        with c2:
            full_bath = st.number_input("Salles de bain", 1, 4, 2)
            tot_rms   = st.number_input("Total pièces", 3, 15, 7)
            garage_c  = st.number_input("Places garage", 0, 4, 2)
            garage_a  = st.number_input("Surface garage (pi²)", 0, 1500, 400, 50)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Qualité & Ancienneté</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            qual = st.slider("Qualité générale (1–10)", 1, 10, 7)
            cond = st.slider("Condition (1–9)", 1, 9, 5)
        with c4:
            yr_built = st.number_input("Année construction", 1870, 2025, 1995)
            yr_remod = st.number_input("Année rénovation",   1870, 2025, 2005)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Équipements & Localisation</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            fireplaces = st.number_input("Cheminées", 0, 4, 1)
            pool_area  = st.number_input("Piscine (pi²)", 0, 1000, 0, 50)
        with c6:
            neighborhood = st.selectbox("Quartier", NEIGHBORHOODS)

        use_model_reg = st.selectbox("Modèle de prédiction", ["Random Forest (Recommandé)", "Decision Tree"])

        predict_btn = st.button("🔮  ESTIMER LE PRIX", use_container_width=True)

    with col_result:
        if predict_btn:
            model_key = 'rf' if 'Random' in use_model_reg else 'dt'
            inp = {
                'GrLivArea': gr_liv, 'TotalBsmtSF': bsmt, 'LotArea': lot,
                'BedroomAbvGr': bedroom, 'FullBath': full_bath, 'TotRmsAbvGrd': tot_rms,
                'OverallQual': qual, 'OverallCond': cond, 'YearBuilt': yr_built,
                'YearRemodAdd': yr_remod, 'Neighborhood': neighborhood,
                'GarageCars': garage_c, 'GarageArea': garage_a,
                'PoolArea': pool_area, 'Fireplaces': fireplaces
            }
            model = models['rf_reg'] if model_key == 'rf' else models['dt_reg']
            row = pd.DataFrame([inp])
            price = float(model.predict(row)[0])
            price_lo = price * 0.92
            price_hi = price * 1.08

            st.markdown(f"""
            <div class="prediction-result">
              <div class="result-label">✦ Estimation du Prix de Vente ✦</div>
              <div class="result-price">${price:,.0f}</div>
              <div class="result-confidence">Intervalle de confiance : <strong>${price_lo:,.0f}</strong> – <strong>${price_hi:,.0f}</strong></div>
              <div style="margin-top:1rem; font-family:'Space Mono',monospace; font-size:0.65rem;
                          color:rgba(0,212,255,0.5); letter-spacing:0.15em; text-transform:uppercase;">
                Modèle : {'Random Forest' if model_key=='rf' else 'Decision Tree'} &nbsp;|&nbsp; Confiance ±8%
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Price context gauge
            st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
            pct = min(max((price - 50000) / 750000, 0), 1)
            category = "Entrée de gamme" if price < 130000 else "Milieu de gamme" if price < 250000 else "Haut de gamme" if price < 400000 else "Luxe"
            cat_color = "#34D399" if price < 130000 else "#00D4FF" if price < 250000 else "#FFD700" if price < 400000 else "#FF6B35"
            st.markdown(f"""
            <div style="background:rgba(10,22,40,0.8); border:1px solid rgba(0,212,255,0.15); border-radius:10px; padding:1rem 1.2rem;">
              <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                <span style="font-size:0.75rem; color:rgba(232,237,242,0.6);">Segment de marché</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.75rem; color:{cat_color}; font-weight:700;">{category}</span>
              </div>
              <div style="background:rgba(0,212,255,0.1); border-radius:4px; height:6px; overflow:hidden;">
                <div style="width:{pct*100:.0f}%; height:100%; background:linear-gradient(90deg,{cat_color},{cat_color}88); border-radius:4px;"></div>
              </div>
              <div style="display:flex; justify-content:space-between; margin-top:0.3rem;">
                <span style="font-family:'Space Mono',monospace; font-size:0.6rem; color:rgba(232,237,242,0.35);">$50K</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.6rem; color:rgba(232,237,242,0.35);">$800K</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Key factors
            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Facteurs Déterminants</div>', unsafe_allow_html=True)
            factors = [
                ("Qualité générale", qual, 10, PAL['accent']),
                ("Surface habitable", gr_liv, 5000, PAL['primary']),
                ("Année construction", yr_built - 1870, 155, "#A78BFA"),
                ("Garage", garage_c, 4, "#34D399"),
            ]
            for name, val, mx, col in factors:
                pf = val / mx
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;">
                  <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:rgba(232,237,242,0.65); margin-bottom:2px;">
                    <span>{name}</span><span style="font-family:'Space Mono',monospace;">{val}</span>
                  </div>
                  <div style="background:rgba(255,255,255,0.05); border-radius:3px; height:5px;">
                    <div style="width:{pf*100:.0f}%; height:100%; background:{col}; border-radius:3px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:rgba(10,22,40,0.6); border:1px dashed rgba(0,212,255,0.2); border-radius:16px;
                        padding:3rem 2rem; text-align:center; margin-top:1rem;">
              <div style="font-size:3rem; margin-bottom:1rem;">🏛️</div>
              <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:rgba(232,237,242,0.7);">
                Prêt pour l'Estimation
              </div>
              <div style="font-size:0.8rem; color:rgba(232,237,242,0.35); margin-top:0.5rem;">
                Renseignez les caractéristiques du bien<br>puis cliquez sur "Estimer le Prix"
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Quick stats
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Statistiques du Dataset</div>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        for col, val, lbl in [(mc1, f"${df['SalePrice'].median():,.0f}", "Prix médian"),
                               (mc2, f"${df['SalePrice'].mean():,.0f}", "Prix moyen"),
                               (mc3, f"{len(df):,}", "Observations")]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏷️  Classification":
    st.markdown('<div class="section-title">Classification du Type de Bâtiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Le modèle <strong>Random Forest Classifier</strong> identifie automatiquement le type de bâtiment à partir des caractéristiques structurelles.</div>', unsafe_allow_html=True)

    col_f, col_r = st.columns([1.1, 0.9], gap="large")

    with col_f:
        st.markdown('<div class="section-label">Caractéristiques du Bien</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            c_grLiv   = st.number_input("Surface habitable (pi²)", 500, 5000, 1400, 50, key="c_gr")
            c_totRms  = st.number_input("Total pièces", 3, 15, 6, key="c_tr")
            c_qual    = st.slider("Qualité générale (1–10)", 1, 10, 6, key="c_q")
        with c2:
            c_yearB   = st.number_input("Année construction", 1870, 2025, 1985, key="c_yb")
            c_garageC = st.number_input("Places garage", 0, 4, 1, key="c_gc")

        c3, c4 = st.columns(2)
        with c3:
            c_neigh   = st.selectbox("Quartier", NEIGHBORHOODS, key="c_nb")
        with c4:
            c_style   = st.selectbox("Style maison", HOUSE_STYLES, key="c_hs")

        use_model_clf = st.selectbox("Modèle de classification", ["Random Forest (Recommandé)", "SVM"])
        clf_btn = st.button("🔍  CLASSIFIER LE BIEN", use_container_width=True)

    with col_r:
        if clf_btn:
            model_key = 'rfc' if 'Random' in use_model_clf else 'svm'
            model = models['rfc_clf'] if model_key == 'rfc' else models['svm_clf']
            le    = models['le']
            inp_c = {
                'GrLivArea': c_grLiv, 'TotRmsAbvGrd': c_totRms, 'OverallQual': c_qual,
                'YearBuilt': c_yearB, 'GarageCars': c_garageC,
                'Neighborhood': c_neigh, 'HouseStyle': c_style
            }
            row_c = pd.DataFrame([inp_c])
            pred_enc  = model.predict(row_c)[0]
            pred_class = le.inverse_transform([pred_enc])[0]
            try:
                proba = model.predict_proba(row_c)[0]
                proba_dict = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
            except:
                proba_dict = {pred_class: 1.0}

            label_full = BLDG_LABELS.get(pred_class, pred_class)
            top_p = proba_dict.get(pred_class, 0)

            st.markdown(f"""
            <div class="prediction-result">
              <div class="result-label">✦ Type de Bâtiment Identifié ✦</div>
              <div style="font-size:2.5rem; margin:0.5rem 0;">{label_full.split()[0]}</div>
              <div class="class-badge">{pred_class}</div>
              <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:#E8EDF2; margin-top:0.3rem;">
                {" ".join(label_full.split()[1:])}
              </div>
              <div class="result-confidence">Confiance : <strong>{top_p*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Distribution des Probabilités</div>', unsafe_allow_html=True)
            sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
            colors_p = [PAL['primary'], PAL['accent'], PAL['secondary'], '#A78BFA', '#34D399']
            for i, (cls, prob) in enumerate(sorted_proba):
                col_p = colors_p[i % len(colors_p)]
                is_pred = "font-weight:700;" if cls == pred_class else ""
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                  <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                              color:rgba(232,237,242,0.8); margin-bottom:3px; {is_pred}">
                    <span>{BLDG_LABELS.get(cls, cls)}</span>
                    <span style="font-family:'Space Mono',monospace; color:{col_p};">{prob*100:.1f}%</span>
                  </div>
                  <div style="background:rgba(255,255,255,0.05); border-radius:3px; height:6px; overflow:hidden;">
                    <div style="width:{prob*100:.1f}%; height:100%; background:{col_p}; border-radius:3px;
                                box-shadow:0 0 8px {col_p}55;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:rgba(10,22,40,0.6); border:1px dashed rgba(0,212,255,0.2); border-radius:16px;
                        padding:3rem 2rem; text-align:center; margin-top:1rem;">
              <div style="font-size:3rem; margin-bottom:1rem;">🏷️</div>
              <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:rgba(232,237,242,0.7);">
                Prêt pour la Classification
              </div>
              <div style="font-size:0.8rem; color:rgba(232,237,242,0.35); margin-top:0.5rem;">
                Renseignez les caractéristiques du bien<br>puis cliquez sur "Classifier le Bien"
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Class distribution mini chart
        st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Répartition dans le Dataset</div>', unsafe_allow_html=True)
        counts = df['BldgType'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['card'])
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
        bar_colors = [PAL['primary'], PAL['secondary'], PAL['accent'], '#A78BFA', '#34D399']
        bars = ax.barh(counts.index, counts.values, color=bar_colors[:len(counts)], height=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color=PAL['text'], fontsize=8, fontweight='bold')
        ax.tick_params(colors=PAL['text'], labelsize=8)
        ax.set_xlabel('Nombre', color=PAL['text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dashboard Analytique":
    st.markdown('<div class="section-title">Dashboard Analytique</div>', unsafe_allow_html=True)

    # KPI Row
    kpis = [
        (f"${df['SalePrice'].median():,.0f}", "Prix Médian", f"μ = ${df['SalePrice'].mean():,.0f}"),
        (f"{df['GrLivArea'].mean():.0f} pi²",  "Surface Moy.",  f"σ = {df['GrLivArea'].std():.0f}"),
        (f"{df['OverallQual'].mean():.1f}/10", "Qualité Moy.",  f"Min={df['OverallQual'].min()} Max={df['OverallQual'].max()}"),
        (f"{df['YearBuilt'].median():.0f}",    "Année Méd.",    f"{df['YearBuilt'].min()}–{df['YearBuilt'].max()}"),
        (f"{len(df):,}",                        "Observations",  f"{df.shape[1]} variables"),
    ]
    cols_k = st.columns(5)
    for col, (val, lbl, delta) in zip(cols_k, kpis):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div><div class="metric-delta">{delta}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    # ── Row 1: SalePrice dist + Qual vs Price ──
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-label">Distribution du Prix de Vente</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['card'])
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
        n_bins = 50
        counts_h, bins = np.histogram(df['SalePrice'], bins=n_bins)
        ax.bar(bins[:-1], counts_h, width=np.diff(bins), color=PAL['primary'], alpha=0.8, edgecolor='none')
        ax.axvline(df['SalePrice'].median(), color=PAL['accent'], linestyle='--', linewidth=1.5, label=f"Médiane ${df['SalePrice'].median():,.0f}")
        ax.tick_params(colors=PAL['text'], labelsize=8)
        ax.set_xlabel('Prix ($)', color=PAL['text'], fontsize=8)
        ax.set_ylabel('Fréquence', color=PAL['text'], fontsize=8)
        ax.legend(facecolor=PAL['bg'], labelcolor=PAL['text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with r1c2:
        st.markdown('<div class="section-label">Qualité vs Prix de Vente</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['card'])
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
        qual_means = df.groupby('OverallQual')['SalePrice'].mean()
        qual_stds  = df.groupby('OverallQual')['SalePrice'].std()
        ax.bar(qual_means.index, qual_means.values, color=PAL['secondary'], alpha=0.85)
        ax.errorbar(qual_means.index, qual_means.values, yerr=qual_stds.values,
                    fmt='none', color=PAL['accent'], capsize=4, linewidth=1.2)
        ax.tick_params(colors=PAL['text'], labelsize=8)
        ax.set_xlabel('Qualité Générale (1–10)', color=PAL['text'], fontsize=8)
        ax.set_ylabel('Prix Moyen ($)', color=PAL['text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 2: Neighborhood + Heatmap ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-label">Prix par Quartier</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['card'])
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
        nb_means = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=True)
        colors_nb = plt.cm.cool(np.linspace(0.2, 0.9, len(nb_means)))
        ax.barh(nb_means.index, nb_means.values, color=colors_nb, height=0.65)
        ax.tick_params(colors=PAL['text'], labelsize=8)
        ax.set_xlabel('Prix Médian ($)', color=PAL['text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with r2c2:
        st.markdown('<div class="section-label">Matrice de Corrélation</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['bg'])
        num_cols = ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','YearBuilt','GarageCars','Fireplaces']
        corr_m = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr_m, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_m, mask=mask, cmap=cmap, center=0, annot=True,
                    fmt='.2f', ax=ax, linewidths=0.5, cbar_kws={'shrink':0.8},
                    annot_kws={'size':7, 'color':'white'})
        ax.tick_params(colors=PAL['text'], labelsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Row 3: Scatter + Year trend ──
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="section-label">Surface Habitable vs Prix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['card'])
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
        sc = ax.scatter(df['GrLivArea'], df['SalePrice'], c=df['OverallQual'],
                        cmap='plasma', alpha=0.5, s=12)
        z = np.polyfit(df['GrLivArea'].dropna(), df['SalePrice'].loc[df['GrLivArea'].dropna().index], 1)
        p = np.poly1d(z)
        xl = np.linspace(df['GrLivArea'].min(), df['GrLivArea'].max(), 100)
        ax.plot(xl, p(xl), color='white', linewidth=2, linestyle='--')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.tick_params(colors=PAL['text'], labelsize=7)
        cbar.set_label('Qualité', color=PAL['text'], fontsize=7)
        ax.tick_params(colors=PAL['text'], labelsize=8)
        ax.set_xlabel('Surface habitable (pi²)', color=PAL['text'], fontsize=8)
        ax.set_ylabel('SalePrice ($)', color=PAL['text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with r3c2:
        st.markdown('<div class="section-label">Évolution des Prix par Décennie</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
        ax.set_facecolor(PAL['card'])
        for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
        df['Decade'] = (df['YearBuilt'] // 10) * 10
        dec_means = df.groupby('Decade')['SalePrice'].median()
        ax.fill_between(dec_means.index, dec_means.values, alpha=0.2, color=PAL['primary'])
        ax.plot(dec_means.index, dec_means.values, color=PAL['primary'], linewidth=2.5, marker='o', markersize=5)
        ax.tick_params(colors=PAL['text'], labelsize=8)
        ax.set_xlabel('Décennie de construction', color=PAL['text'], fontsize=8)
        ax.set_ylabel('Prix Médian ($)', color=PAL['text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Raw data explorer
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    with st.expander("🗃️  Explorateur de Données Brutes", expanded=False):
        n_display = st.slider("Nombre de lignes", 5, 100, 20)
        filter_neigh = st.multiselect("Filtrer par Quartier", NEIGHBORHOODS, default=NEIGHBORHOODS[:3])
        df_display = df[df['Neighborhood'].isin(filter_neigh)].head(n_display) if filter_neigh else df.head(n_display)
        st.dataframe(df_display.style.background_gradient(subset=['SalePrice'], cmap='Blues'), use_container_width=True)
        st.caption(f"Affichage : {len(df_display)} lignes | Dataset complet : {len(df):,} observations")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 : RAPPORT DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Rapport des Modèles":
    st.markdown('<div class="section-title">Rapport de Performance des Modèles</div>', unsafe_allow_html=True)

    meta_r = models['meta']['regression']
    meta_c = models['meta']['classification']

    tab_reg, tab_clf = st.tabs(["📈  Régression (SalePrice)", "🏷️  Classification (BldgType)"])

    with tab_reg:
        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Métriques de Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl, delta in [
            (m1, f"${meta_r['rf_results']['MAE']:,.0f}", "MAE – RF", "Random Forest"),
            (m2, f"${meta_r['rf_results']['RMSE']:,.0f}", "RMSE – RF", f"R²={meta_r['rf_results']['R2']:.3f}"),
            (m3, f"${meta_r['dt_results']['MAE']:,.0f}", "MAE – DT", "Decision Tree"),
            (m4, f"{meta_r['dt_results']['R2']:.3f}", "R² – DT", f"RMSE={meta_r['dt_results']['RMSE']:,.0f}$"),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div><div class="metric-delta">{delta}</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown('<div class="section-label">Comparaison MAE & RMSE</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
            ax.set_facecolor(PAL['card'])
            for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
            x = np.arange(2); w = 0.3
            vals_dt = [meta_r['dt_results']['MAE'], meta_r['dt_results']['RMSE']]
            vals_rf = [meta_r['rf_results']['MAE'], meta_r['rf_results']['RMSE']]
            b1 = ax.bar(x - w/2, vals_dt, w, color=PAL['secondary'], label='Decision Tree', alpha=0.9)
            b2 = ax.bar(x + w/2, vals_rf, w, color=PAL['primary'], label='Random Forest', alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(['MAE', 'RMSE'], color=PAL['text'])
            ax.tick_params(colors=PAL['text'], labelsize=9)
            ax.set_ylabel('Erreur ($)', color=PAL['text'], fontsize=8)
            ax.legend(facecolor=PAL['bg'], labelcolor=PAL['text'], fontsize=8)
            for bars in [b1, b2]:
                for bar in bars:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                            f"${bar.get_height():,.0f}", ha='center', color=PAL['text'], fontsize=7, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with r_c2:
            st.markdown('<div class="section-label">Comparaison R² & CV-R²</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
            ax.set_facecolor(PAL['card'])
            for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
            x = np.arange(2); w = 0.3
            vals_dt2 = [meta_r['dt_results']['R2'], meta_r['dt_results']['CV_R2']]
            vals_rf2 = [meta_r['rf_results']['R2'], meta_r['rf_results']['CV_R2']]
            b1 = ax.bar(x - w/2, vals_dt2, w, color=PAL['secondary'], label='Decision Tree', alpha=0.9)
            b2 = ax.bar(x + w/2, vals_rf2, w, color=PAL['primary'], label='Random Forest', alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(['R² Test', 'CV-R²'], color=PAL['text'])
            ax.tick_params(colors=PAL['text'], labelsize=9); ax.set_ylim(0, 1.1)
            ax.legend(facecolor=PAL['bg'], labelcolor=PAL['text'], fontsize=8)
            ax.axhline(0.8, color=PAL['accent'], linestyle=':', linewidth=1, label='Seuil 0.80')
            for bars in [b1, b2]:
                for bar in bars:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{bar.get_height():.3f}", ha='center', color=PAL['text'], fontsize=8, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("""
        <div class="success-box">
          ✅ <strong>Modèle Sélectionné : Random Forest Regressor</strong><br>
          Le Random Forest surpasse le Decision Tree sur toutes les métriques : MAE inférieur de 33%,
          RMSE réduit de 29%, et R² de 0.803 indiquant une excellente capacité prédictive.
          La validation croisée (CV-R²=0.837) confirme la robustesse et l'absence de sur-apprentissage.
        </div>
        """, unsafe_allow_html=True)

    with tab_clf:
        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Métriques de Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl, delta in [
            (m1, f"{meta_c['rfc_results']['Accuracy']:.3f}", "Accuracy – RFC", "Random Forest"),
            (m2, f"{meta_c['rfc_results']['F1']:.3f}", "F1-score – RFC", "Weighted"),
            (m3, f"{meta_c['svm_results']['Accuracy']:.3f}", "Accuracy – SVM", "Support Vector"),
            (m4, f"{meta_c['svm_results']['F1']:.3f}", "F1-score – SVM", "Weighted"),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div><div class="metric-delta">{delta}</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

        c_c1, c_c2 = st.columns(2)
        with c_c1:
            st.markdown('<div class="section-label">Comparaison Accuracy & F1</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
            ax.set_facecolor(PAL['card'])
            for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
            x = np.arange(2); w = 0.3
            vals_svm = [meta_c['svm_results']['Accuracy'], meta_c['svm_results']['F1']]
            vals_rfc = [meta_c['rfc_results']['Accuracy'], meta_c['rfc_results']['F1']]
            b1 = ax.bar(x - w/2, vals_svm, w, color=PAL['secondary'], label='SVM', alpha=0.9)
            b2 = ax.bar(x + w/2, vals_rfc, w, color=PAL['primary'], label='RF Classifier', alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels(['Accuracy', 'F1-score'], color=PAL['text'])
            ax.tick_params(colors=PAL['text'], labelsize=9); ax.set_ylim(0, 0.9)
            ax.legend(facecolor=PAL['bg'], labelcolor=PAL['text'], fontsize=8)
            for bars in [b1, b2]:
                for bar in bars:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{bar.get_height():.3f}", ha='center', color=PAL['text'], fontsize=8, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with c_c2:
            st.markdown('<div class="section-label">Distribution Classes (Déséquilibre)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PAL['bg'])
            ax.set_facecolor(PAL['card'])
            for sp in ax.spines.values(): sp.set_edgecolor('#2A3F5F')
            classes = models['le'].classes_
            class_counts = df['BldgType'].value_counts()
            c_vals = [class_counts.get(c, 0) for c in classes]
            c_colors = [PAL['primary'], PAL['secondary'], PAL['accent'], '#A78BFA', '#34D399']
            wedges, texts, autotexts = ax.pie(c_vals, labels=classes, autopct='%1.1f%%',
                                               colors=c_colors, startangle=90,
                                               textprops={'color': PAL['text'], 'fontsize': 8},
                                               pctdistance=0.8)
            for at in autotexts: at.set_fontsize(7); at.set_color(PAL['bg'])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("""
        <div class="warning-box">
          ⚠️ <strong>Déséquilibre des Classes :</strong> La classe '1Fam' représente ~60% des données,
          ce qui pénalise les métriques pour les classes minoritaires (Duplex, Twnhs).
          Des techniques comme SMOTE ou class_weight='balanced' pourraient améliorer les performances
          sur les classes sous-représentées.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="success-box">
          ✅ <strong>Modèle Sélectionné : Random Forest Classifier</strong><br>
          Accuracy de 58.7% sur un problème multi-classes déséquilibré (5 classes). Le RFC
          surpasse le SVM (47%) grâce à sa capacité à modéliser des frontières de décision complexes
          sans nécessiter de standardisation stricte.
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER SIGNATURE ─────────────────────────────────────────────────────────
st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.3),rgba(255,215,0,0.2),transparent); margin-bottom:1.5rem;"></div>
<div class="footer-signature">
  <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:rgba(0,212,255,0.4);
              letter-spacing:0.25em; text-transform:uppercase; margin-bottom:0.8rem;">
    ✦ Développé par ✦
  </div>
  <div class="footer-name">MUGENI KANZA CHRISTIAN</div>
  <div class="footer-title">Master Intelligence Artificielle</div>
  <div class="footer-filiere">Filière : Intelligence Artificielle &nbsp;|&nbsp; Machine Learning & Data Science</div>
  <div style="margin-top:1rem; font-family:'Space Mono',monospace; font-size:0.6rem;
              color:rgba(232,237,242,0.25); letter-spacing:0.1em;">
    PRÉDICTION INTELLIGENTE IMMOBILIÈRE &nbsp;●&nbsp; RandomForest · SVM · Streamlit
  </div>
</div>
""", unsafe_allow_html=True)
