import streamlit as st
from PIL import Image
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import json
import zipfile
import os
from io import BytesIO
from models.identification_model import load_model, make_prediction, create_image_with_bboxes
from models.segmentation_model import load_segmentation_model, segment_and_save_objects
from models.text_extraction_model import load_easyocr_model, extract_text
from models.summarization_model import load_summarization_model, generate_summary
from utils.preprocessing import preprocess_image, preprocess_text
from utils.postprocessing import postprocess_segmentation, postprocess_identification
from utils.data_mapping import map_data_to_objects, save_data_mapping

# Load models
detection_model = load_model()
segmentation_model = load_segmentation_model()
ocr_reader = load_easyocr_model()
summarizer = load_summarization_model()

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]

# Placeholder descriptions (you might want to generate or retrieve these descriptions based on your requirements)
descriptions = [f"Description of {category}" for category in categories]

# Streamlit app
# Streamlit app
st.title("AI-based Image Analysis 🤖🔍📊")

upload = st.file_uploader(label="Upload your Image Here:", type=["png", "jpeg", "jpg"])

if upload:
    try:
        img = Image.open(upload)
        img = img.convert('RGB')  # Ensure RGB format

        preprocessed_img = preprocess_image(img)

       # Object Detection
        detection_prediction = make_prediction(detection_model, img)

       # Visualization with bounding boxes
        image_with_boxes = create_image_with_bboxes(np.array(img), detection_prediction)

       # Display the image with bounding boxes
        st.image(image_with_boxes, caption="Detected Objects with Bounding Boxes", use_column_width=True)

        # Post-process predictions
        detection_prediction = postprocess_identification(detection_prediction, categories)

        # Extract object data (labels and bounding boxes) to pass to mapping function
        detected_objects = []
        for i, box in enumerate(detection_prediction['boxes']):
            detected_objects.append({
                "id": i,
                "category": detection_prediction['labels'][i],
                "bounding_box": box  # No need for .tolist()
            })

        # Segmentation
        master_id, metadata = segment_and_save_objects(segmentation_model, img, "output")

        # Text Extraction
        extracted_text = extract_text(ocr_reader, img)
        processed_text = [preprocess_text(text) for text in extracted_text]

        # Summarization
        summaries = generate_summary(summarizer, processed_text)

        # Mapping and Saving
        mapped_data = map_data_to_objects(metadata, processed_text, summaries, descriptions)

        # Integrate detected object categories into the mapped data
        for i, obj in enumerate(mapped_data['objects']):
            if i < len(detected_objects):
                obj['category'] = detected_objects[i]['category']
                obj['bounding_box'] = detected_objects[i]['bounding_box']

        # Save the mapping data as JSON
        save_data_mapping(mapped_data, "output/mapping.json")

        # Create a metadata.txt file
        metadata_text = "Master ID: {}\n".format(master_id)
        for data in metadata:
            metadata_text += "Object ID: {}\n".format(data['object_id'])
            metadata_text += "Bounding Box: {}\n\n".format(data['bbox'])
        
        with open("output/metadata.txt", "w") as f:
            f.write(metadata_text)

        # Zip all outputs (JSON, metadata, segmented images)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.write("output/mapping.json", arcname="mapping.json")
            zip_file.write("output/metadata.txt", arcname="metadata.txt")
            
            for img_name in os.listdir("output"):
                if img_name.endswith(".png"):
                    zip_file.write(os.path.join("output", img_name), arcname=img_name)

        zip_buffer.seek(0)

        st.success("Processing complete!")
        # Download button for the ZIP file
        st.download_button(
            label="Download Analysis",
            data=zip_buffer,
            file_name="output.zip",
            mime="application/zip"
        )

        st.write("Files saved as output.zip")

    except Exception as e:
        st.error(f"An error occurred: {e}")
