# Wildlife Detection with MegaDetector

This repository contains code for identifying and detecting the presence of animals in images using MegaDetector, a machine learning model designed for wildlife detection. The model is capable of recognizing animals, people, and vehicles in images, making it ideal for ecological research and conservation projects. The detections from the Megadetector model have been passed as inputs to the YOLOv5 model, which has been fine-tuned to carry out identification of species. With this tool, researchers can automate the process of analyzing large datasets of wildlife images, streamlining efforts in biodiversity monitoring and wildlife tracking. This code provides an end-to-end pipeline for image processing, from loading datasets to visualizing detections.

To get started with this project, first clone the repository to your local machine using the following command:

`git clone https://github.com/Abhik-Biswas/uwin-megadetector.git`

### Steps to run the Megadetector Model
  1. Install the megadetector package

     `pip install megadetector`
     
  2. The above is supposed to install the dependencies as well. If not, the following also needs to be installed:
     
     `pip3 install torch torchvision torchaudio streamlit`

  3. Then run the following script:

     `python -m streamlit run md_app,py`

     Supply the Source Directory, the Destination Directory, and the name of the output `json` file. The Megadetector model will run inference on the the images in the directory, and store all the bounding box coordinates 
     which detected the presence of an object with confidence greater than or equal to the set threshold. 

