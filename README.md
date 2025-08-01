## Defective-Product-Detector
This project is a image classification built using TensorFlow and MobileNetV2 to detect casting defects in products. 

## Approach & Assumptions

- **Pre-trained model used**: `MobileNetV2` trained on ImageNet for transfer learning.
- **Assumption**: Dataset is well-separated into two classes: `def_front` and `ok_front`, with equal preprocessing and image sizes.

## File Overview
| File                | Description                                   |
| ------------------- | --------------------------------------------- |
| `Script.ipynb`      | Model training pipeline                       |
| `utils.py`          | DefectDetector class for prediction           |
| `defect_model.h5`   | Trained MobileNetV2 model weights             |
| `requirements.txt`  | Required Python packages                      |
| `test_images/`      | Sample images for quick prediction            |

## Install required packages
pip install -r requirements.txt


## Setup Instructions
```bash
git clone https://github.com/BeninASolomon/Defective-Product-Detector.git
cd Defective-Product-Detector
```

from utils import DefectDetector
# Load model
model = DefectDetector("defect_model.h5")
# Predict on a single image
result = model.predict("test_images/sample.jpg")
# Output
print(result) 


