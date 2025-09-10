import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template_string, request, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Caption Generator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; background: #f7f7f7; }
    .container { max-width: 600px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
    h2 { text-align: center; }
    .image-preview img { max-width: 100%; border-radius: 8px; }
    .caption-box { margin-top: 20px; padding: 15px; background: #f0f0f0; border-left: 6px solid #007bff; border-radius: 6px; }
    .caption-box h3 { margin-top: 0; }
    input[type=file] { margin-bottom: 10px; }
    input[type=submit] { background: #007bff; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
    input[type=submit]:hover { background: #0056b3; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Image Caption Generator</h2>
    <form method=post enctype=multipart/form-data>
      <input type=file name=image>
      <input type=submit value='Generate Caption'>
    </form>
    {% if image_url %}
      <div class="image-preview">
        <h3>Uploaded Image:</h3>
        <img src="{{ image_url }}">
      </div>
    {% endif %}
    {% if caption %}
      <div class="caption-box">
        <h3>Generated Caption:</h3>
        <p>“{{ caption }}”</p>
      </div>
    {% endif %}
    {% if error %}
      <div class="caption-box" style="border-left:6px solid red;">
        <h3>Error:</h3>
        <p style="color:red;">{{ error }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
'''

# Load VGG16 model for feature extraction
vgg16_model = VGG16()
vgg16_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)

# Load your trained captioning model
caption_model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

def get_word_from_index(index, tokenizer):
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

def predict_caption(model, image_features, tokenizer, max_caption_length):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        if predicted_word is None:
            break
        caption += " " + predicted_word
        if predicted_word == "endseq":
            break
    return caption.replace("startseq", "").replace("endseq", "").strip()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_url = None
    error = None
    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No file part."
            return render_template_string(HTML_TEMPLATE, caption=caption, image_url=image_url, error=error)
        file = request.files['image']
        if file.filename == '':
            error = "No selected file."
            return render_template_string(HTML_TEMPLATE, caption=caption, image_url=image_url, error=error)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join('static', filename)
            os.makedirs('static', exist_ok=True)
            file.save(img_path)
            image_url = url_for('static', filename=filename)
            try:
                image = Image.open(img_path).convert('RGB').resize((224, 224))
                image_array = img_to_array(image)
                image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
                image_array = preprocess_input(image_array)
                image_feature = vgg16_model.predict(image_array, verbose=0)
                max_caption_length = 34  # Set this to your trained value
                caption = predict_caption(caption_model, image_feature, tokenizer, max_caption_length)
            except Exception as e:
                error = f"Prediction error: {e}"
        else:
            error = "Invalid file type. Please upload a PNG or JPG image."
    return render_template_string(HTML_TEMPLATE, caption=caption, image_url=image_url, error=error)

if __name__ == '__main__':
    app.run(debug=True)