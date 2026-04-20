# Skin lesion classifier
A 7-class skin lesion classifier fine-tuned on ResNet18 using transfer learning, trained on the HAM10000 dermoscopic image dataset.

## Dataset
- **Source:** [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Size:** 10,015 dermoscopic images
- **Classes:** akiec, bcc, bkl, df, mel, nv, vasc
- **Split:** 80% train / 10% validation / 10% test

Limitations:
-
- The model misses 85% of melanomas, so it is not very useful to detect it.
- Classes are imbalanced in the amount of training data, so rare classes result in poor performance.
- Model is trained on professional dermoscope images, not regular phone photos

Tech stack:
-
Python, PyTorch, ResNet18, Torchvision, Gradio, Pandas, Scikit-learn, Seaborn

Instructions:
-

**Requirements:** Python 3.13+, CUDA-compatible GPU recommended

**1. Clone the repository**
```bash
git clone https://github.com/AlexandrSdrv/skin-lesion-classifier.git
cd skin-lesion-classifier
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**4. Download the model**

Download `skin_lesion_model.pth` and place it in the `data/` folder.

**5. Run the app**
```bash
python app.py
```
Then open **http://127.0.0.1:7860** in your browser.
