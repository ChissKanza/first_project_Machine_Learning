# 🏛️ Prédiction Intelligente Immobilière — Immo Predictor

**MUGENI KANZA CHRISTIAN | Master IA | Intelligence Artificielle**

---

## 📁 Structure du Projet

```
CHRISTIAN_MUGENI_KANZA/
├── app.py                          # Application Streamlit principale
├── utils.py                        # Utilitaires et fonctions auxiliaires
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
├── MUGENI_KANZA_CHRISTIAN_Notebook.ipynb   # Notebook ML complet
├── Rapport_Academique_MUGENI_KANZA_CHRISTIAN.pdf
├── models/
│   ├── dt_regressor.pkl            # Decision Tree Regressor (pipeline)
│   ├── rf_regressor.pkl            # Random Forest Regressor (pipeline)
│   ├── svm_classifier.pkl          # SVM Classifier (pipeline)
│   ├── rfc_classifier.pkl          # RF Classifier (pipeline)
│   ├── label_encoder.pkl           # LabelEncoder BldgType
│   └── metadata.json               # Métriques et paramètres
└── data/
    ├── housing_data.csv             # Dataset (1500 observations)
    └── plots/                       # Visualisations EDA
```

---

## 🚀 Lancement Local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'application
streamlit run app.py
```

L'application sera disponible sur http://localhost:8501

---

## ☁️ Déploiement Streamlit Cloud

1. **Pousser sur GitHub :**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Immo Predictor"
   git remote add origin https://github.com/VOTRE_USER/immo-predictor.git
   git push -u origin main
   ```

2. **Déployer :**
   - Aller sur [share.streamlit.io](https://share.streamlit.io)
   - Se connecter avec GitHub
   - Sélectionner le dépôt → Branch: `main` → File: `app.py`
   - Cliquer **Deploy**

3. **Configuration :** Aucune variable d'environnement requise.

---

## 📊 Performances des Modèles

| Tâche | Modèle | Métrique | Score |
|-------|--------|----------|-------|
| Régression | Random Forest | R² | **0.803** |
| Régression | Random Forest | MAE | **~14 800 $** |
| Régression | Decision Tree | R² | 0.604 |
| Classification | RF Classifier | Accuracy | **58.7%** |
| Classification | SVM | Accuracy | 47.0% |

---

## 🔧 Technologies

- **Python 3.10+**
- **pandas, numpy** — Manipulation de données
- **matplotlib, seaborn** — Visualisations
- **scikit-learn** — ML (Pipeline, GridSearchCV, RF, SVM)
- **streamlit** — Application web
- **joblib** — Sérialisation des modèles

---

*© MUGENI KANZA CHRISTIAN — Master Intelligence Artificielle*
