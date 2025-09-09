import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = 'model/mnist_cnn_model.keras'


# --- 1. Load the trained model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from the .keras file."""
    model = tf.keras.models.load_model(MODEL_PATH) # type: ignore
    return model

model = load_model()

# --- 2. Set up the Streamlit app layout ---
st.title("Handwritten Digit Recognizer")
st.markdown("Draw a digit in the box below, then click 'Predict'.")

col1, col2 = st.columns([1, 1])

with col1:
    # Create a canvas component to allow the user to draw
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300, width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    predict_button = st.button("Predict Digit")  # Prediction button

# Option to display debug images
st.sidebar.markdown("### Debugging Options")
show_debug_images = st.sidebar.checkbox("Show debug images")


# --- 3. Process the drawn image and make a prediction ---
if predict_button and canvas_result.image_data is not None:
    # Convert the canvas data to a NumPy array
    img = canvas_result.image_data.astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize the image (scale pixels to 0-1)
    normalized = resized / 255.0

    # Reshape the image to fit the model's input format (28, 28, 1)
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Get the model's prediction
    prediction_probs = model.predict(reshaped)
    prediction = np.argmax(prediction_probs)
    
    # Get the confidence score for the prediction
    confidence = np.max(prediction_probs) * 100

    # --- 4. Display the prediction result ---
    st.markdown(f"### Prediction: **{prediction}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # --- 5. Debugging Section (Conditional Display) ---
    if show_debug_images:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Debug Images")

        # Display the original drawn image (grayscale version)
        st.sidebar.image(gray, caption="Drawn Image (Grayscale)", width='stretch')

        # Display the resized and normalized image that goes into the model
        fig_resized, ax_resized = plt.subplots(figsize=(2, 2))
        ax_resized.imshow(normalized.reshape(28, 28), cmap='gray')
        ax_resized.axis('off')
        st.sidebar.pyplot(fig_resized)
        st.sidebar.caption("Preprocessed Image (28x28, Normalized)")