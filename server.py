from flask import Flask, render_template, current_app, request
import json

app = Flask(__name__)

@app.route("/")
def hello_world():
    return current_app.send_static_file("index.html")



@app.route("/predict", methods = ['POST'])
def predict():
    data = request.get_json()
    
    response = {'HELLO': 'from python', 'prediction': 3}
    response = json.dumps(response)
    
    return response