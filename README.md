An AI Pipeline for Image Segmentation and Object Analysis

This project implements an AI pipeline for detecting objects and extracting text from images, using models like Faster R-CNN, image segmentation, and OCR. The app also summarizes detected objects and text, providing a comprehensive analysis of the uploaded image.
Features
Object Detection: Uses Faster R-CNN to detect and classify objects in an image. Image Segmentation: Segments detected objects, saving each as a separate image. Text Extraction: Extracts text from images using OCR (EasyOCR). Summarization: Summarizes the detected objects and extracted text. Visualization: Displays segmented objects and associated data, with a summary table.
Installation
Clone the repository:

git clone https://github.com/Divyaprem/wasserstoff-AiInternTask

Navigate to the project directory:

cd project_root

Create a virtual environment:

python -m venv .venv

Activate the virtual environment:

On Windows: bash .venv\Scripts\activate
On macOS/Linux: bash source .venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Usage Run the Streamlit app:

streamlit run app.py

Upload an image in the app interface to analyze it.
