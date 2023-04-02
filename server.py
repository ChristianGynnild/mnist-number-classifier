from flask import Flask, render_template, current_app, request
import json
from PIL import Image
import PIL
import numpy as np
import ai

app = Flask(__name__)

model = ai.load_model()


@app.route("/")
def hello_world():
    return current_app.send_static_file("index.html")



@app.route("/api/predict", methods = ['POST'])
def predict():
    data = request.get_json()

    channels_amount = data['channels_amount']
    columns_amount = data['columns_amount']
    rows_amount = data['rows_amount']

    pixels = data['pixels']
    pixels = np.array(list(pixels.values())).reshape((columns_amount, rows_amount, channels_amount)).astype('uint8')
    image = Image.fromarray(pixels, mode="RGBA")
    image = image.resize((28, 28), Image.NEAREST)
    
    number = ai.predict(model, image)
    print(number)

    image.convert('RGB')

    response = {'HELLO': 'from python', 'prediction': 3}
    response = json.dumps(response)
    
    return response