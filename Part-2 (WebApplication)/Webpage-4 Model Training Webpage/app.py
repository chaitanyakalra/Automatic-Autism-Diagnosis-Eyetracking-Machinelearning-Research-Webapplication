import os
import shutil
import glob
import cv2
import random
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
from flask_cors import CORS
import tensorflow as tf
import io

##Preprocessing functions 
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

def grayscale_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def preprocess_images(images):
    gray_images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (224, 224)) for img in images]
    return [grayscale_to_rgb(img) for img in gray_images]

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainedModels")
app.secret_key = 'secret-key'

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Route for uploading dataset
@app.route("/upload", methods=["POST"])
def upload_dataset():
    if request.method == "POST":
        dataset = request.files.get('dataset')
        if dataset:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            dataset.save(os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.zip'))
            flash("Zip file successfully uploaded.")
        return redirect(url_for("home"))
    
# Route for uploading a custom model
@app.route("/upload_model", methods=["POST"])
def upload_model():
    if request.method == "POST":
        model = request.files.get('model_upload')
        if model:
            model.save(os.path.join(models_path, model.filename))
            flash("Model successfully uploaded.")
        return redirect(url_for("home"))

    

# Route for handling model selection and training
@app.route('/', methods=['GET', 'POST'])
def home():
    # Get all .h5 files from the TrainedModels directory
    models = [os.path.basename(f) for f in glob.glob(os.path.join(models_path, '*.h5'))]
    
    if not models:
        flash("No models found. Please upload a model first.")
    
    if request.method == 'POST':
        epochs = request.form.get('epochs')
        model_name = request.form.get('model')
        
        if epochs and model_name:
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.zip')
            if os.path.isfile(dataset_path):
                retrain_model(int(epochs), os.path.join(models_path, model_name))
            else:
                flash("Dataset not found. Please upload the dataset before Starting Training.")
    
    return render_template('index.html', models=models)


     
#Function for retraining 

def retrain_model(epochs, model_filename):
    # Load the selected model
    model = load_model(model_filename)

    # Extract the dataset
    import zipfile
    with zipfile.ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.zip'), 'r') as zip_ref:
        zip_ref.extractall(app.config['UPLOAD_FOLDER'])

    # Load and preprocess images
    dataset_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')

    # Automatically detect the folder names for the two classes
    subfolders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    class1_folder, class2_folder = subfolders[0], subfolders[1]

    images_class1 = load_images_from_folder(class1_folder)
    images_class2 = load_images_from_folder(class2_folder)

    preprocessed_images_class1 = preprocess_images(images_class1)
    preprocessed_images_class2 = preprocess_images(images_class2)

    X = np.array(preprocessed_images_class1 + preprocessed_images_class2)
    y = np.array([0] * len(preprocessed_images_class1) + [1] * len(preprocessed_images_class2))

    # Split the data into train and test sets (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Retrain the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_split=0.2)

    # Save the retrained model in the TrainedModels directory
    retrained_model_path = os.path.join(models_path, 'model_retrained.h5')
    model.save(retrained_model_path)

    # Display training accuracy, epoch number, and loss
    for epoch, acc, loss in zip(range(1, epochs + 1), history.history['accuracy'], history.history['loss']):
        print(f"Epoch: {epoch}, Accuracy: {acc:.2f}, Loss: {loss:.2f}")

    # Call clean_upload_folder() after the training process is complete
    clean_upload_folder()

    # Re-enable the "Start Training" button and update the flashed message
    flash("Training has finished. You can now download the retrained model.")

def clean_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# new routine For downloading the retrained model and then delete it from local mempry after downloading using threading
from flask import send_from_directory

import threading
import time

def delete_file(filepath):
    while True:
        try:
            os.remove(filepath)
            print(f"Local copy of {filepath} deleted.")
            break
        except Exception as e:
            print(f"Failed to delete local copy of {filepath}. Retrying...")
            time.sleep(1)

@app.route('/download', methods=['GET'])
def download():
    try:
        retrained_model_path = os.path.join(models_path, 'model_retrained.h5')
        response = send_from_directory(directory=models_path, path='model_retrained.h5', as_attachment=True)

        t = threading.Thread(target=delete_file, args=(retrained_model_path,))
        t.start()

        return response
    except Exception as e:
        print(f"Failed to download retrained model. Reason: {e}")
        return redirect(url_for('home'))



def file_exists(filepath):
    return os.path.isfile(os.path.join(models_path, filepath))


app.jinja_env.globals['file_exists'] = file_exists
if __name__ == '__main__':

    app.run(debug=True, use_reloader=False)
