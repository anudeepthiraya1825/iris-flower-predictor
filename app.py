import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Predictor")

@st.cache_data
def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, iris

model, iris = train_model()

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = iris.target_names
    st.success(f"Predicted Species: 🌼 {species[prediction[0]].capitalize()}")
