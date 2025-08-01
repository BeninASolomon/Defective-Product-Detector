## Defective-Product-Detector
This project is a image classification built using TensorFlow and MobileNetV2 to detect casting defects in products. 

## Approach & Assumptions
Model: Pre-trained MobileNetV2 (on ImageNet) used with transfer learning.
Assumptions:
- Dataset is cleanly split into two classes: def_front and ok_front.
- All images are preprocessed and resized to the same dimension (128x128).
- defect_model.h5 is the trained model file for predictions.

## Repository Structure
| File                           | Description                                       |
| -------------------------------| --------------------------------------------------|
| `Script.ipynb`                 | Jupyter Notebook with training pipeline           |
| `utils.py`                     | DefectDetector class for prediction               |
| `defect_model.h5`              | Trained MobileNetV2 model weights                 |
| `requirements.txt`             | Required Python packages                          |
| `def.jpeg & non_def.jpeg`      | Sample images for quick prediction                |
| `example.ipynb`                | Demo notebook to validate predictions using model |

## Installation
```bash
git clone https://github.com/BeninASolomon/Defective-Product-Detector.git
cd Defective-Product-Detector
pip install -r requirements.txt
```

## Usage (Class-based)
from utils import DefectDetector
model = DefectDetector("defect_model.h5")
result = model.predict("test_images/sample.jpg")
print(result)

## Evaluation Metrics
The model is evaluated using:
Accuracy
F1 Score
Precision
Recall


