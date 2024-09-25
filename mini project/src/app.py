from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
from model import create_model, get_num_classes
import json

app = Flask(__name__)

# Load your trained model
model_path = 'best_model_fold_0.pth'
num_classes = get_num_classes(model_path)
model = create_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load remedies data
with open('remedies.json', 'r') as f:
    remedies_data = json.load(f)

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define your class names
class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 
               'Septoria Leaf Spot', 'Yellow Leaf Curl Virus']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            probability = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        predicted_class = class_names[predicted.item()]
        remedies_info = remedies_data.get(predicted_class, {})
            
        return jsonify({
            'class': predicted_class,
            'probability': probability[predicted.item()].item(),
            'symptoms': remedies_info.get('symptoms', 'No information available'),
            'remedies': remedies_info.get('remedies', 'No information available'),
            'prevention': remedies_info.get('prevention', 'No information available')
        })

if __name__ == '__main__':
    app.run(debug=True)