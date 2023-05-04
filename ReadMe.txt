This Folder Contains all files related to Part 1 of my project 
5.2  Download Requirements 
The Following packages and applications need to be installed:
	- Jupyter Notebook
	- Python > 3.10 and the following packages:
			Open CV (cv2), numpy, sklearn, tensorflow, matplotlib, seaborn, 
MobileNetV3, ResNet50, VVG16
JAVA Script
PYTHON FLASK 

{5.3 User Manual 
Accessing the Research materials 
All files for the research part of the project are located in the "Part-1 (Research)" folder of the submission. The folder contains the following subfolders and files:
Dataset: This folder contains two subfolders:
"original": Contains the original dataset of images.
"Augmented5x": Contains the images produced using the original dataset through augmentation.
DataProcessingFolder: Contains Python scripts in Jupyter Lab notebook format for preprocessing the dataset, including augmentation, contrast increase, cropping, and splitting the dataset.
Framework 1 - PCA Models: Contains Jupyter Lab notebook files for the PCA models.
Framework 2 - TransferLearning Models: Contains Jupyter Lab notebook files for the Transfer Learning models.
Framework 3 - CNNmain Model: Contains Jupyter Lab notebook files for the CNN model.
SavedModels: Contains all the final models obtained from the research in Part 1, in .h5 and .pkl formats.

All Jupyter notebooks have outputs already showing, including the results from testing the models, how the dataset was processed and trained, the hyperparameters used, and graphs and other illustrations.

Accessing the different webpages of the Web Application 

All files related to the development of the web application "Automatic Autism Diagnosis using Eye Tracking and Machine Learning" are present in the "Part-2(Web Application)" folder. The folder contains the following subfolders and files:

Webpage-1 EyeTracking Webpage: Contains the main.html file for the eye tracking webpage. To access this webpage, double-click the "1_EyeTrackingWebpage" HTML shortcut or navigate to the main.html file in the subfolder. Follow these steps to use the webpage:
Click "Begin Tracking" and grant camera permission. A video preview of the webcam should appear on the top left side of the screen, and a moving red dot should indicate where the user is looking.
Click "Calibrate" to calibrate the eye tracking. A window overlay with red dots will appear. Click each button until it turns green, then close the overlay with the cross button in the top left corner.
Click "Begin Recording" to start saving eye tracking data in a CSV file.
Use the "Toggle Video Preview" button to enable or disable the video preview.
Adjust the "Sampling Minimum Delay" slider from 0ms to 5000ms to introduce a sampling rate delay. Click "Apply" to apply the changes.
Use the "Pause Tracking" and "Resume Tracking" buttons to pause and resume eye tracking.
Click "Save" to stop recording and download the CSV file.
Use the dropdown list to choose from regression models for eye tracking. Click "Apply" to apply the selected model.
Click "Visualize Eye Tracking" in the top right corner to navigate to the next page.

 Webpage-2 EyeTracking Visualisation Webpage: Contains the index.html file for the eye tracking visualization webpage. To access this webpage, double-click the "2_EyeTrackingVisualiserWebpage" HTML shortcut or navigate to the index.html file in the subfolder. Follow these steps to use the webpage:
Click "Choose File" to upload a CSV file of eye tracking data. The CSV should have the format: X, Y, Timestamp.
Set the smoothing level by choosing from the list and clicking "Apply."
The "Original" container shows the eye tracking visualization without smoothing.
Scroll down to see the "Smoothed" container, which shows the eye tracking visualization with smoothing.
Click "Record Eye Tracking" in the top right corner to navigate back to the previous webpage.

Webpage-3 Prediction Webpage: Contains the app.py file for the prediction webpage. To access this webpage, double-click the "3_PredictionWebpage" Python shortcut or navigate to the app.py file in the subfolder. A console window should appear, starting the Python Flask local server. Navigate to the URL shown in the console window (usually http://127.0.0.1:5000/). Follow these steps to use the webpage:
Use the "Select a Model" dropdown list to choose a pretrained model.
(Optional) Click "Upload your Own Model" to upload a deep learning model.
(Optional) Set class names by filling in the "Class 0 Name" and "Class 1 Name" text fields.
Click "Upload an Image" to upload the eye tracking visualization image for prediction.
Click "Run Prediction" to run the prediction. The predicted results will appear at the bottom of the webpage.

Webpage-4 Model Training Webpage: Contains the app.py file for the model training webpage. To access this webpage, double-click the "4_ModelTrainingWebpage-A" Python shortcut or navigate to the app.py file in the subfolder. A console window should appear, starting the Python Flask local server. Make sure to close the console from the previous webpage to avoid conflicting servers on the same URL. Navigate to the URL shown in the console window (usually http://127.0.0.1:5000/). Follow these steps to use the webpage:
In the "Select Dataset" section, click "Choose File" and select a dataset in ZIP format. Click "Upload" to upload the file. The ZIP file should contain a folder named "dataset" with two subfolders containing images of their respective classes.
(Optional) Click "Choose File" and select a model in .h5 format to upload your own model. Click "Upload" to upload the file.
Choose a model from the list of models in the "Select Model" section.
Set the number of epochs for training in the "Epochs" number field.
Click "Start Training" to begin the training process. The progress and training metrics can be viewed in the console window of the Python Flask app.
After the training process is complete, a "Download Retrained Model" button will appear to download the retrained model.
