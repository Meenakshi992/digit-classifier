import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import plotly.graph_objects as go
# Part-1 
# Trains & saves 'mnist_cnn_enhanced.h5'
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Load and normalize dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 2. Build the CNN Architecture with BN and Dropout
model = models.Sequential([
    # First Conv Block
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Conv Block
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(256),  # Increased capacity for richer feature matching
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

# 3. Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training high-stability CNN with Batch Normalization...")
model.fit(x_train, y_train, epochs=6, batch_size=64, validation_split=0.1)

# 4. Export model
model.save("mnist_cnn_enhanced.h5")
print("Model saved successfully as 'mnist_cnn_enhanced.h5'!")

st.set_page_config(
    page_title="CNN Digit Classifier Diagnostic",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MNIST Digit Classifier App")
st.write(
    "Draw a single digit (0–9) below. The application captures your canvas data and performs live evaluation across the model pipeline.")


# ----------------------------------------------------
# 1. "mnist_cnn_enhanced.h5" Model Attachment Pipeline
# ----------------------------------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("mnist_cnn_enhanced.h5")
        return model, False
    except Exception:
        return None, True


model, is_demo_mode = load_cnn_model()

with st.sidebar:
    st.header("Model running status check & Setup")
    if is_demo_mode:
        st.warning("Running in DEMO mode")
    else:
        st.success("CNN Model compiled and active!")

    st.markdown("---")
    st.markdown("Canvas Options")
    stroke_width = st.slider("Brush Thickness", min_value=6, max_value=36, value=18)
    show_tensor = st.checkbox("Show Model Input Tensor", value=True)

# ----------------------------------------------------
# 2. Two Column Layout
# ----------------------------------------------------
col_canvas, col_metrics = st.columns([1, 1.2])

with col_canvas:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
               stroke_width=16, 
               stroke_color="#FFFFFF",
               background_color="#000000",
               width=280,
               height=280,
               drawing_mode="freedraw",
               key="uncached_canvas",
    )

# ----------------------------------------------------
# 3. Preprocessing Operations
# ----------------------------------------------------
if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
    raw_rgba = canvas_result.image_data
    gray_img = cv2.cvtColor(raw_rgba.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Advanced Preprocessing: Center the digit to mimic MNIST distribution bounding box logic
    # Find bounding box around drawn contours
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        # Crop the active region containing the digit
        cropped = gray_img[y:y + h, x:x + w]

        # Add padding to maintain an aspect-ratio friendly bounding box matrix
        pad = max(w, h) // 4
        padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        resized_img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        resized_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)

    normalized_img = resized_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.expand_dims(normalized_img, axis=-1), axis=0)

    # --- Prediction Matrix Calculations ---
    if not is_demo_mode:
        prediction_probabilities = model.predict(input_tensor, verbose=0)[0]
    else:
        # Dynamic deterministic mock matrix based on structural pixel distribution
        pixel_seed = int(np.sum(resized_img)) % 10
        mock_logits = np.array([np.sin(i + pixel_seed) + 1.1 if i == pixel_seed else 0.1 for i in range(10)])
        prediction_probabilities = mock_logits / np.sum(mock_logits)

    predicted_digit = np.argmax(prediction_probabilities)
    confidence_score = prediction_probabilities[predicted_digit] * 100

    # ----------------------------------------------------
    # 4. Render Interface Analytics
    # ----------------------------------------------------
    with col_metrics:
        st.subheader("Model Digit Prediction Analysis")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="Predicted Digit", value=f"{predicted_digit}")
        m_col2.metric(label="Confidence", value=f"{confidence_score:.1f}%")

        classes = [f"Digit {i}" for i in range(10)]
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=prediction_probabilities,
                text=[f"{p * 100:.1f}%" for p in prediction_probabilities],
                textposition='auto',
                marker_color=["#CC4100" if i == predicted_digit else '#636EFA' for i in range(10)]
            )
        ])
        fig.update_layout(
            yaxis=dict(title="Probability", range=[0, 1.05], tickformat=".0%"),
            xaxis=dict(title="Classes"),
            margin=dict(l=20, r=20, t=10, b=10),
            height=260,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    if show_tensor:
        with st.sidebar:
            st.markdown("---")
            st.markdown("# Predicted Matrix (28x28)")
            st.image(resized_img, caption="What the CNN actually evaluates", width=160)
else:
    with col_metrics:
        st.info("Write a single digit inside the box canvas to view the output.")
