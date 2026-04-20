import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_descriptions = {
    'akiec': 'Actinic Keratosis / Bowen\'s Disease',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi (Mole)',
    'vasc': 'Vascular Lesion'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)

checkpoint = torch.load('data/skin_lesion_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
    
    results = {class_descriptions[class_names[i]]: float(probs[i]) for i in range(7)}
    return results

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload Skin Lesion Image'),
    outputs=gr.Label(num_top_classes=7, label='Prediction'),
    title='🔬 Skin Lesion Classifier',
    description='Upload a dermoscopic image to classify the skin lesion type. ⚠️ This is for educational purposes only — not a medical diagnosis tool.',
)

demo.launch()