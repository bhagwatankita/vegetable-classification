import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the model you trained
model = load_model('vegetable_model.h5')

# The 15 labels in alphabetical order (standard for ImageDataGenerator)
LABELS = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # 1. Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 2. Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 3. Predict
        preds = model.predict(x)
        idx = np.argmax(preds)
        
        # 4. SEND TO HTML (Match the names label and img_path)
        return render_template('prediction.html', 
                               label=LABELS[idx], 
                               img_path=file.filename) # Just the filename

if __name__ == '__main__':
    app.run(debug=True)