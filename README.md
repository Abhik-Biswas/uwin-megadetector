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
     
![VID32432-00553](https://github.com/user-attachments/assets/8ac688d9-40c0-4ccd-83d4-14b855fd85ad)

     

### Segregation into Empty vs. Non-Empty Images
  1. The notebook `Directory Builder.ipynb` helps create a copy of the directory structure (with empty folders) of the source folder in the destination specified.
  2. Once the folder is created, open the `Empty Vs Non Empty.ipynb` notebook. Modify the file path of the `json` file, the source folder and the destination folder, and execute the cell. This would create two sub- 
     directories - `empty_images` and `non_empty_images`, where the segregated images would be stored.

### Creating the data
  1. The data creation process, to be fed into the YOLOv5 model is outlined here: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#23-organize-directories

     This process has been closely mimicked, and the dataset has been created using the contents in the notebook: `^YOLO-Data-Creation.*\.ipynb`

     
  2. Once the data has been created, the training can be started using:

     `python train.py --img 640 --batch 16 --epochs 3 --data UWIN.yaml --weights yolov5s.pt`

     This downloads pre-trained weights on the COCO Dataset, and uses them as a starting point. A custom filepath containing a more specialized set of weights can also be provided as the input to the weights argument. 
     
  4. Once training has been carried out, run inference using the trained weights using:

     `python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images --save_csv True`

     Again, a custom filepath can be specificied for both the model weights, and the image directory, on which you want to run the inference on. The `save_csv` argument, if set to `True`, saves the predictions, along with the image names in a `csv` file.

![results](https://github.com/user-attachments/assets/846ff39a-4717-4a6c-8a52-df7ed91969db)

![F1_curve](https://github.com/user-attachments/assets/de72f055-3178-489f-8f4d-3db0505f3c3d)

![P_curve](https://github.com/user-attachments/assets/6211d80e-788c-4838-9d42-274470dba611)

![PR_curve](https://github.com/user-attachments/assets/e29fa3e2-0aa7-4726-b809-221f89d697a8)

![R_curve](https://github.com/user-attachments/assets/d313ef8f-00fc-49ce-be62-d1eee3fe93a1)

![val_batch2_labels](https://github.com/user-attachments/assets/b8350ece-0edc-455b-969d-49f062bfa274)

![val_batch2_pred](https://github.com/user-attachments/assets/8bd071f8-cecf-495c-87a9-6da1a194f913)



  6. 



