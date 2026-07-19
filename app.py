# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load the trained model
# model = tf.keras.models.load_model("cats_dogs_model.keras")

# st.title("🐱 Cats vs Dogs Image Classifier")

# uploaded_file = st.file_uploader(
#     "Upload a Cat or Dog Image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file)

#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     image = image.resize((128, 128))

#     img_array = np.array(image)

#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)

#     score = prediction[0][0]

#     if score >= 0.5:
#         st.success(f"🐶 Dog ({score*100:.2f}% confidence)")
#     else:
#         st.success(f"🐱 Cat ({(1-score)*100:.2f}% confidence)")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_model.keras")

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🐱 Cats vs Dogs Image Classifier")
st.write(
    "Upload an image of a **Cat** or a **Dog** and let the CNN predict it."
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📖 About")

st.sidebar.info("""
This application uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.

### Model Details
- Image Size: 128 × 128
- Classes:
  - 🐱 Cat
  - 🐶 Dog
- Framework:
  - TensorFlow
  - Streamlit

Developed as an Image Classification Project.
""")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Convert to RGB (handles grayscale images)
    image = image.convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    # Resize
    resized_image = image.resize((128, 128))

    # Convert to numpy array
    img_array = np.array(resized_image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Predicting..."):

        prediction = model.predict(img_array, verbose=0)

    score = prediction[0][0]

    st.divider()

    st.subheader("Prediction Result")

    if score >= 0.5:
        confidence = score * 100

        st.success("🐶 Dog")

    else:
        confidence = (1 - score) * 100

        st.success("🐱 Cat")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

st.divider()

st.caption("Made with ❤️ using TensorFlow, Keras and Streamlit")