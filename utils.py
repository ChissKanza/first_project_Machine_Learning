"""
utils.py - Utilitaires pour Immo Predictor
MUGENI KANZA CHRISTIAN | Master IA | Intelligence Artificielle
"""

import joblib
import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_models():
    """Charge tous les modèles et artefacts."""
    models = {}
    try:
        models['dt_reg'] = joblib.load(os.path.join(MODELS_DIR, 'dt_regressor.pkl'))
        models['rf_reg'] = joblib.load(os.path.join(MODELS_DIR, 'rf_regressor.pkl'))
        models['svm_clf'] = joblib.load(os.path.join(MODELS_DIR, 'svm_classifier.pkl'))
        models['rfc_clf'] = joblib.load(os.path.join(MODELS_DIR, 'rfc_classifier.pkl'))
        models['label_encoder'] = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
            models['metadata'] = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des modèles : {e}")
    return models


def predict_price(models, input_data: dict, use_model: str = 'rf') -> float:
    """
    Prédit le prix d'un bien immobilier.
    input_data: dict avec les features
    use_model: 'rf' ou 'dt'
    """
    meta = models['metadata']['regression']
    all_features = meta['features_num'] + meta['features_cat']
    df = pd.DataFrame([{f: input_data.get(f, np.nan) for f in all_features}])
    model = models['rf_reg'] if use_model == 'rf' else models['dt_reg']
    return float(model.predict(df)[0])


def predict_bldg_type(models, input_data: dict, use_model: str = 'rfc') -> tuple:
    """
    Prédit le type de bâtiment.
    Retourne (classe_prédite, probabilités_dict)
    """
    meta = models['metadata']['classification']
    all_features = meta['features_num'] + meta['features_cat']
    df = pd.DataFrame([{f: input_data.get(f, np.nan) for f in all_features}])
    model = models['rfc_clf'] if use_model == 'rfc' else models['svm_clf']
    le = models['label_encoder']
    pred_enc = model.predict(df)[0]
    pred_class = le.inverse_transform([pred_enc])[0]
    try:
        proba = model.predict_proba(df)[0]
        proba_dict = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    except Exception:
        proba_dict = {pred_class: 1.0}
    return pred_class, proba_dict


def format_price(price: float) -> str:
    return f"${price:,.0f}"


NEIGHBORHOODS = ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst',
                 'NridgHt', 'Gilbert', 'Sawyer', 'NWAmes', 'Mitchel']

HOUSE_STYLES = ['1Story', '2Story', '1.5Fin', 'SFoyer', 'SLvl']

BLDG_TYPE_LABELS = {
    '1Fam': '🏡 Maison individuelle (1Fam)',
    '2fmCon': '🏘️ Maison bi-familiale (2fmCon)',
    'Duplex': '🏠 Duplex',
    'TwnhsE': '🏙️ Townhouse (extrémité)',
    'Twnhs': '🏙️ Townhouse (intérieur)'
}
