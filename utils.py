# import necessary Packages 
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


class DefectDetector:
    # Here load Model and class 
    def __init__(self, model_path):
        model = tf.keras.models.load_model(model_path)
        class_names = ['defect', 'ok']
        self.model, self.class_names = model, class_names
    
    # here given image to load, normalize and predict 
    # after that print class and and the prediction score 
    def predict(self, image_path):
        img = image.load_img(image_path, target_size=(128, 128))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)[0]
        idx = np.argmax(prediction)
        return {
            "status": self.class_names[idx],
            "confidence_score": round(float(prediction[idx]), 4)
        }
