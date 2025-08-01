## Defective-Product-Detector
This project is a image classification built using TensorFlow and MobileNetV2 to detect casting defects in products. 

## Approach & Assumptions
Model: Pre-trained MobileNetV2 (on ImageNet) used with transfer learning.
Assumptions:
- Dataset is cleanly split into two classes: def_front and ok_front.
- All images are preprocessed and resized to the same dimension (128x128).
- defect_model.h5 is the trained model file for predictions.
  
## Dataset
- [Download Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
  
## Repository Structure
| File                           | Description                                       |
| -------------------------------| --------------------------------------------------|
| `Script.ipynb`                 | Jupyter Notebook with training pipeline           |
| `utils.py`                     | DefectDetector class for prediction               |
| `requirements.txt`             | Required Python packages                          |
| `def.jpeg & non_def.jpeg`      | Sample images for quick prediction                |
| `example.ipynb`                | Demo notebook to validate predictions using model |


## Model Weights
- [Download defect_model.h5](https://drive.google.com/uc?export=download&id=14rJlQLPYgbjiD3dqxWVxi4xjLsf4zQBl)
(because it big to upload in github)

## Installation
```bash
# Clone the repository
git clone https://github.com/BeninASolomon/Defective-Product-Detector.git
cd Defective-Product-Detector

# Install required packages
pip install -r requirements.txt

# runs notebooks
pip install notebook
jupyter notebook
run example.ipynb
```

## Usage (Class-based)
```
from utils import DefectDetector
import tensorflow as tf
import os
import gdown
#load the model from google drive because it big to upload in git
model_path = "defect_model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=14rJlQLPYgbjiD3dqxWVxi4xjLsf4zQBl"
    gdown.download(url, model_path, quiet=False)
    
model = DefectDetector(model_path)
result = model.predict("def.jpeg")
print(result)
```

## Evaluation Metrics
The model is evaluated using:
- Accuracy
- F1 Score
- Precision
- Recall


