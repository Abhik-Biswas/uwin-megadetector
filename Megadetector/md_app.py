import streamlit as st
import os
import json
import cv2
from tqdm import tqdm
from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, detections, threshold):
    for detection in detections:
        if detection['conf'] > threshold:
            bbox = detection['bbox']
            category = detection['category']
            conf = detection['conf']
            height, width, _ = image.shape
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int((bbox[0] + bbox[2]) * width)
            y2 = int((bbox[1] + bbox[3]) * height)
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            # Draw the label
            label = f'Category: {category}, Conf: {conf:.2f}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            y1_label = y1 + label_size[1] + 10 if y1 + label_size[1] + 10 < y2 else y1 - 10
            x1_label = x1 + 10 if x1 + label_size[0] + 20 < x2 else x1 - label_size[0] - 20
            cv2.rectangle(image, (x1_label, y1_label - label_size[1] - 10), (x1_label + label_size[0] + 20, y1_label + base_line - 10), (0, 0, 255), cv2.FILLED)
            cv2.putText(image, label, (x1_label + 10, y1_label - 7), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

# Function to run MegaDetector with progress tracking
def run_detector_with_progress(detector_name, image_file_names):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    stop_button = st.button('Stop')
    total_images = len(image_file_names)
    for i, image_path in enumerate(image_file_names):
        if stop_button:
            st.write("Processing stopped by user.")
            break
        result = load_and_run_detector_batch(detector_name, [image_path])
        results.extend(result)
        progress_percent = (i + 1) / total_images
        progress_bar.progress(progress_percent)
        status_text.text(f"Processing: {progress_percent * 100:.2f}%")
    return results

# Streamlit app
st.title("MegaDetector Batch Processing")

# Input directory for images
image_folder = st.text_input("Enter the directory of images:", "")
image_folder = fr"{image_folder}"
output_directory = st.text_input("Enter the directory to save processed images:", "")
output_directory = fr"{output_directory}"
confidence_threshold = st.slider("Set confidence threshold:", 0.0, 1.0, 0.2)
output_file = st.text_input("Enter the output JSON file path:", "enter file name to save to (JSON)")

if st.button("Run MegaDetector"):
    if image_folder and output_directory and output_file:
        # Find images in the directory
        image_file_names = path_utils.find_images(image_folder, recursive=True)
        
        # Run MegaDetector with progress tracking
        st.write("Running MegaDetector...")
        results = run_detector_with_progress('MDV5A', image_file_names)

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Process each image and save the results
        st.write("Drawing bounding boxes on images...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        # stop_button = st.button("Stop")
        for i in tqdm(range(len(results))):
            image_path = results[i]['file']
            image = cv2.imread(image_path)
            draw_bounding_boxes(image, results[i]['detections'], confidence_threshold)
            # Save the processed image to the output directory
            output_path = os.path.join(output_directory, os.path.basename(image_path))
            cv2.imwrite(output_path, image)
            progress_percent = (i + 1) / len(results)
            progress_bar.progress(progress_percent)
            status_text.text(f"Processing: {progress_percent * 100:.2f}%")
        
        st.success(f"Processed images saved to '{output_directory}'")
        
        # Save results to JSON file
        with open(output_file, 'w') as file:
            json.dump(results, file, indent=4)
        st.success(f"Results saved to '{output_file}'")
    else:
        st.error("Please provide valid directories and output file path.")
