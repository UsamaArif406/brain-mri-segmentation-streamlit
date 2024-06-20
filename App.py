import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2

plt.style.use("ggplot")

# Define functions for metrics (dice coefficients, IoU, etc.)
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

# Load the pre-trained model
model = load_model("unet_brain_mri_seg.hdf5", custom_objects={
        'dice_coef_loss': dice_coefficients_loss, 'iou': iou, 'dice_coef': dice_coefficients})

# Define image dimensions
im_height = 256
im_width = 256

# Function to preprocess uploaded images
def preprocess_image(image):
    img_resized = cv2.resize(image, (im_height, im_width))
    img_normalized = img_resized / 255.0  # Normalize image
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    return img_expanded

# Function to perform inference and visualize results
def predict_image(image):
    pred_img = model.predict(image)
    return pred_img

# Streamlit UI for the application
def main():
    st.title("Brain MRI Segmentation App")

    # Sidebar for input file upload
    upload_mode = st.sidebar.radio("Select upload mode:", ("Single Image", "Multiple Images"))

    if upload_mode == "Single Image":
        file = st.sidebar.file_uploader("Upload file", type=["png", "jpg"])
        if file:
            st.header("Original Image:")
            st.image(file, width=256, caption="Original Image")
            try:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                processed_image = preprocess_image(image)
                predicted_image = predict_image(processed_image)
                st.header("Predicted Image:")
                st.image(predicted_image[0], width=256, caption="Predicted Image")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif upload_mode == "Multiple Images":
        files = st.sidebar.file_uploader("Upload multiple files", type=["png", "jpg"], accept_multiple_files=True)
        if files:
            for idx, file in enumerate(files):
                st.header(f"Original Image {idx+1}:")
                st.image(file, width=256, caption=f"Original Image {idx+1}")
                try:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    processed_image = preprocess_image(image)
                    predicted_image = predict_image(processed_image)
                    st.header(f"Predicted Image {idx+1}:")
                    st.image(predicted_image[0], width=256, caption=f"Predicted Image {idx+1}")
                except Exception as e:
                    st.error(f"Error processing image {idx+1}: {e}")

if __name__ == "__main__":
    main()
