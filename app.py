import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained CNN model
model = tf.keras.models.load_model('food_model.h5')

from util import classify, set_background


set_background('bgs/bg5.png')

# Define the food classes
food_classes = [
    "Carrot Cake", "Mussels", "Lobster Bisque", "Chocolate Cake", "Crab Cakes",
    "Fried Rice", "Ice Cream", "Garlic Bread", "Pizza", "Samosa", "Onion Rings"
]

# Streamlit app
st.title("Food Image Classification ")

st.write("Your Classses are Carrot Cake, Mussels, Lobster Bisque, Chocolate Cake, Crab Cakes,Fried Rice, Ice Cream, Garlic Bread, Pizza, Samosa, Onion Rings")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img1 = Image.open(uploaded_file).convert('RGB')
    img = img1.resize((64, 64))  # Assuming your model expects 64x64 images
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = food_classes[np.argmax(prediction)]

    # Display results
    st.image(img1, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction:")
    st.write(f"Class: {predicted_class}")
    st.write(f"Confidence: {prediction[0][np.argmax(prediction)]:.2%}")
