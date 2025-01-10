import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os

# Configuration de la page
st.set_page_config(
    page_title="Food Classification & Calorie Estimation",
    page_icon="🍔",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover { background-color: #ff6b6b; }
    .statistics-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .calorie-card {
        background-color: #e8f4ea;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    h1, h2, h3 { color: #1f1f1f; }
    .description-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions de chargement
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('food_classification_model.keras')

@st.cache_resource
def load_label_encoder():
    return joblib.load('label_encoder.pkl')

@st.cache_resource
def load_normal_params():
    return load_normal_params_all_sheets('density.xls')

def load_normal_params_all_sheets(excel_path: str) -> dict:
    xls = pd.ExcelFile(excel_path)
    df_list = []
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        df_list.append(df_sheet)
    
    df = pd.concat(df_list, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]
    
    grouped = df.groupby('type')['calorie (cl)']
    mean_std_dict = {}
    for food_type, subgrp in grouped:
        mu = subgrp.mean()
        sigma = subgrp.std()
        mean_std_dict[food_type] = (mu, sigma)
    
    return mean_std_dict

def normal_calorie_score(mu: float, sigma: float) -> float:
    if (mu is None) or np.isnan(mu):
        return None
    if (sigma is None) or np.isnan(sigma) or (sigma == 0.0):
        return mu
    return float(np.random.normal(mu, sigma))

# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Fonction pour la prédiction avec estimation des calories
def predict_class_and_calories(image, model, label_encoder, normal_params):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class_index])
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Estimation des calories
    mu_sigma = normal_params.get(predicted_class.lower())
    if mu_sigma is not None:
        mu, sigma = mu_sigma
        calories = normal_calorie_score(mu, sigma)
    else:
        calories = None
        
    return predicted_class, confidence, calories

# Chargement des modèles et paramètres
try:
    model = load_model()
    label_encoder = load_label_encoder()
    normal_params = load_normal_params()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles: {str(e)}")
    normal_params = {}

# Navigation
pages = {
    "🏠 Accueil": "home",
    "🔍 Classification & Calories": "classification",
    "📈 Perspectives Futures": "future"
}

# Sidebar pour la navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Aller à", list(pages.keys()))

# Page d'accueil
if pages[selection] == "home":
    st.title("🍽️ Classification Alimentaire & Estimation des Calories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="description-text">
        <h2>À propos du projet</h2>
        Notre système utilise l'intelligence artificielle pour :
        <ul>
            <li>Identifier automatiquement les aliments à partir d'images</li>
            <li>Estimer les calories avec une approche statistique avancée</li>
            <li>Fournir des informations nutritionnelles détaillées</li>
            <li>Aide à la planification des repas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="statistics-card">
            <h3>Caractéristiques du Système</h3>
            <ul>
                <li>Modèle CNN haute précision</li>
                <li>Estimation calorique basée sur la distribution normale</li>
                <li>Base de données nutritionnelle étendue</li>
                <li>Interface utilisateur intuitive</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Page de classification
elif pages[selection] == "classification":
    st.title("🔍 Classification & Estimation des Calories")
    
    uploaded_file = st.file_uploader(
        "Téléversez une image d'aliment",
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés : JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image téléversée", use_column_width=True)
        
        with col2:
            with st.spinner("Analyse en cours..."):
                predicted_class, confidence, calories = predict_class_and_calories(
                    image, model, label_encoder, normal_params
                )
                
                # Affichage des résultats
                st.markdown("""
                <div class="calorie-card">
                    <h3>Résultats de l'analyse</h3>
                """, unsafe_allow_html=True)
                
                # Jauge de confiance
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': "Confiance de la prédiction"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4b4b"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig)
                
                # Résultats
                st.success(f"Aliment identifié : {predicted_class}")
                if calories is not None:
                    st.info(f"Calories estimées : {calories:.1f} kcal/100g")
                    
                    # Graphique des calories
                    df_calories = pd.DataFrame({
                        'Catégorie': ['Calories estimées', 'Moyenne'],
                        'Calories': [calories, normal_params[predicted_class.lower()][0]]
                    })
                    
                    fig_calories = px.bar(
                        df_calories,
                        x='Catégorie',
                        y='Calories',
                        title='Comparaison avec la moyenne',
                        color='Catégorie'
                    )
                    st.plotly_chart(fig_calories)
                    
                    # Recommandations
                    st.markdown("""
                    <div class="statistics-card">
                        <h4>Informations nutritionnelles</h4>
                        <ul>
                            <li>Ces valeurs sont des estimations basées sur des moyennes statistiques</li>
                            <li>Les calories réelles peuvent varier selon la préparation</li>
                            <li>Consultez un professionnel pour des conseils personnalisés</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Impossible d'estimer les calories pour cet aliment.")

# Page des perspectives futures
elif pages[selection] == "future":
    st.title("📈 Perspectives Futures")
    
    st.markdown("""
    <div class="description-text">
    <h2>Améliorations prévues</h2>
    
    <h3>1. Intelligence Artificielle</h3>
    <ul>
        <li>Amélioration de la précision du modèle de classification</li>
        <li>Raffinement du modèle d'estimation des calories</li>
        <li>Intégration de la détection de portions</li>
    </ul>
    
    <h3>2. Fonctionnalités</h3>
    <ul>
        <li>Journal alimentaire personnalisé</li>
        <li>Calcul des macronutriments</li>
        <li>Suggestions de repas équilibrés</li>
    </ul>
    
    <h3>3. Interface Utilisateur</h3>
    <ul>
        <li>Application mobile native</li>
        <li>Support multilingue</li>
        <li>Mode hors-ligne</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulaire de feedback
    st.markdown("### Votre avis compte !")
    with st.form("feedback_form"):
        st.write("Aidez-nous à améliorer l'application")
        feedback = st.text_area("Suggestions d'amélioration")
        submitted = st.form_submit_button("Envoyer")
        if submitted:
            st.success("Merci pour votre feedback !")

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: gray;'>© {datetime.now().year} Food Classification & Calorie Estimation. Tous droits réservés.</p>",
    unsafe_allow_html=True
)