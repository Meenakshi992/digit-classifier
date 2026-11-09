
# Part 2: Main uncached interactive Streamlit frontend
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

st.set_page_config(page_title="CNN Classifier (No Cache)", layout="centered")
st.title("🚀 High-Precision Character Identifier")
st.write("Draw a single digit. Caching has been completely disabled for real-time model asset evaluation.")

# 1. Load the model directly without any @st.cache_resource decorators
def load_cnn_model():
    try:
        return tf.keras.models.load_model("mnist_cnn_enhanced.h5")
    except Exception as e:
        st.error("Could not find 'mnist_cnn_enhanced.h5'. Please run your train_model.py file first.")
        return None

model = load_cnn_model()

col1, col2 = st.columns(2)

with col1:
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

with col2:
    st.subheader("Analysis Metrics")
    
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
        # Image extraction & Grayscale reduction
        raw_img = canvas_result.image_data.astype(np.uint8)
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2GRAY)
        
        # Extract bounding contour box around user stroke
        bbox = cv2.boundingRect(gray_img)
        x, y, w, h = bbox
        
        if w > 0 and h > 0:
            cropped = gray_img[y:y+h, x:x+w]
            pad_size = int(max(w, h) * 0.35)
            padded = cv2.copyMakeBorder(cropped, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
            smoothed = cv2.GaussianBlur(padded, (3, 3), 0)
        else:
            smoothed = gray_img

        # Standardize dimensions & normalize data tensor structures
        resized_img = cv2.resize(smoothed, (28, 28), interpolation=cv2.INTER_AREA)
        normalized_img = resized_img.astype("float32") / 255.0
        input_tensor = np.expand_dims(np.expand_dims(normalized_img, axis=-1), axis=0)
        
        # Execute Real-time Inference Pass
        if model is not None:
            predictions = model.predict(input_tensor, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
            # Display metrics dashboard
            st.metric(label="Predicted Character", value=str(predicted_class))
            st.progress(int(confidence))
            st.write(f"**Certainty Metric:** {confidence:.2f}%")
            
            with st.expander("Live Debug Matrix (28x28 feed)"):
                st.image(resized_img, width=140, caption="Normalized Input Channel")
    else:
        st.info("Awaiting structural strokes. Use your cursor to sketch inside the square window.")
