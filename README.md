# Brain MRI Segmentation using U-Net with Streamlit

This repository contains the implementation of a Brain MRI Segmentation application using a U-Net architecture, deployed with Streamlit. The application allows users to upload MRI images and get segmented outputs indicating different regions of the brain.

## Overview

Brain MRI segmentation is a crucial task in medical image analysis, aimed at identifying different regions of the brain from MRI scans. This project leverages a U-Net architecture for performing segmentation and provides an interactive interface using Streamlit for ease of use.

## Project Structure

```plaintext
brain-mri-segmentation-streamlit/
│
├── Accuracy Graph.png            # Graph showing the model accuracy over epochs
├── App.py                        # Streamlit application script
├── Loss Graph.png                # Graph showing the model loss over epochs
├── Notebook.ipynb                # Jupyter Notebook with model training and evaluation
├── Unet_Architecture.png         # Image of the U-Net architecture used
├── requirements.txt              # Required dependencies for the project
└── README.md                     # Project documentation
```
# Installation
Prerequisites
Ensure you have Python 3.7 or higher installed.

## Install the required packages using pip:
pip install -r requirements.txt

#Model Architecture
The model used for brain MRI segmentation is based on the 
U-Net architecture, which is particularly well-suited for biomedical image segmentation.



# Results
The following graphs show the model's accuracy and loss over the training epochs:

Accuracy Graph:
Loss Graph:

# Acknowledgements
This project builds upon various open-source tools and libraries in the Python ecosystem, including but not limited to:

TensorFlow/Keras for model building and training
Streamlit for creating the web application
NumPy and Pandas for data handling
Matplotlib for plotting graphs

# Usage
Ensure all dependencies are installed (see Installation section).

## Run the Streamlit application:
streamlit run App.py

