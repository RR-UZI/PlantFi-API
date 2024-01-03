import torch
import torchvision
from flask import Flask, render_template, request, jsonify
from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

from torchvision.models import resnet18
import torch.nn.functional as F
import re


app = Flask(__name__)

filename = '/Users/uzi/Desktop/8vo_Semestre/Programacion_Para_Internet/PlantFi/results/resnet18_weights_best_acc.tar' # pre-trained model path
use_gpu = True  # load weights on the gpu
model = resnet18(num_classes=1081) # 1081 classes in Pl@ntNet-300K

load_model(model, filename, False)

model.eval()

@app.route('/' , methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def hello_word():
    return render_template('about.html')

@app.route('/about', methods=['POST'])
def predict():
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image provided.'}), 400
    
    imagefile= request.files['imagefile']
    image_path = "static/img" + imagefile.filename
    print(image_path)
    imagefile.save(image_path)
    
        # Open and preprocess the image
    image = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Pass the preprocessed image through your PyTorch model to obtain the model's output.
    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = F.softmax(output, dim=1)  # Apply softmax

    _, predicted_class = probabilities.max(1)
    
    # Get the probability associated with the predicted class
    predicted_probability = torch.gather(probabilities, 1, predicted_class.view(-1, 1))

    # Convert the probability to percentage
    predicted_percentage = predicted_probability.item() * 100
    
    with open('order_DataSet.json', 'r') as json_file:
        data = json.load(json_file)
    
    if str(predicted_class.item()) in data:
        classification = data[str(predicted_class.item())]['plant_name']
    
    else:
        print("Predicted class not found in the JSON data.")

    # return render_template('about.html', prediction=classification, image_path=image_path)
    return jsonify({'prediction': classification}, {'Porcentaje de precision:' : predicted_percentage})


if __name__ == '__main__':
    app.run(port=3000, debug=True)


    



  # Get the predicted class index





