import streamlit as st
import numpy as np
import joblib

# ============================
# Chargement du modèle
# ============================
@st.cache_resource
def load_model():
    data = joblib.load("modele_manual_vf1.pkl")
    return data["model"], data["scaler"], data["features"]

model, scaler, features = load_model()

# ============================
# Titre
# ============================
st.title("Intrusion Detection System")

st.write("Détection d'attaque réseau par Machine Learning")

# ============================
# Saisie utilisateur
# ============================
st.subheader("Saisie des caractéristiques")

user_input = []

for feature in features:
    value = st.number_input(
        feature,
        value=0.0
    )
    user_input.append(value)

# ============================
# Prédiction
# ============================
if st.button("Predict"):
    X = np.array(user_input).reshape(1, -1)

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    st.subheader("Résultat")

    if prediction == 0:
        st.write("Trafic normal")
    else:
        st.write("Attaque détectée")
