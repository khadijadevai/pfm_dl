import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 🎨 Personnaliser l’arrière-plan avec une couleur simple
st.markdown(
    """
    <style>
    .stApp {
        background-color: #EEDDC8; /* marron bébé clair */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📦 Charger le modèle
model = load_model("mon_modele_fine_tune.h5")

# 🔍 Fonction de prédiction
def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Moderne" if prediction >= 0.5 else "Historique"
    return label, prediction

# 🖼 Interface Streamlit
st.title(" Prédiction : Bâtiment Moderne ou Historique")

uploaded_file = st.file_uploader(" Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l’image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Image téléchargée", width=300)

    # Prédiction
    label, prediction = predict_single_image(uploaded_file)
    # Résultat : Afficher label + probabilité associée à cette classe
    if label == "Moderne":
        proba = prediction
    else:
        proba = 1 - prediction

    # Résultat : Afficher label + probabilité
    st.markdown(f"### Prédiction : **{label}**")
    st.markdown(f"Probabilité : **{proba:.2f}**")

    # Visualisation graphique
    fig, ax = plt.subplots()
    ax.barh(["Historique", "Moderne"], [1 - prediction, prediction], color=["#DDA853", "#183B4E"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité")
    ax.set_title("Probabilité par classe")
    st.pyplot(fig)
