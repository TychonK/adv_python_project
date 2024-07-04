# 🧑‍🎓 Age Prediction by Face Photo Keras Model 📷

## 📝 Description
This project is a written in Python neural network. The project explores different variations of Keras CNNs learning from a dataset of 10,000 face photos to predict the age of a person based on their face image.

## 🌟 Best Model
The best model identified in this project is **`age_model_3.keras`**.

## 🛠️ Installation
To run this project, you'll need the following dependencies:
- **Python**
- **TensorFlow 2.16.1**
- **Keras 3.3.3**
- **OpenCV (cv2)**
- **NumPy**
- **Pandas**
- **Matplotlib**

You can install these dependencies using pip:
```bash
pip install tensorflow==2.16.1 keras==3.3.3 opencv-python numpy pandas matplotlib
```

## 🚀 Usage
To use this project, follow these steps:

1. Clone the repository.
2. Ensure you have the required dependencies installed.
3. Run the cells in the provided Jupyter notebook from top to bottom.
   - **❗ Note:** Be careful when loading models in different cells as they use the same variable name `loaded_model` and will override each other.

There is a picture of the owner of the repository that can be used to test the model's age prediction **`test.jpg`**. The owner is actually 16 years old in that picture.

## 🖥️ Example
```python
# Example of loading and using a model
from keras.models import load_model

# Load the best model
model = load_model('<PATH_TO_MODEL>/age_model_3.keras')

# Load and preprocess the image
import cv2
import numpy as np

img = cv2.imread("<IMAGE_PATH>")  
img = cv2.resize(img, (200, 200))  
img = img.astype('float32') / 255.0  


img = np.expand_dims(img, axis=0)

# Check the shape
print('Image shape:', img.shape)

predicted_age = loaded_model.predict(img)

print('Predicted age:', predicted_age[0][0])
```

Will give out the age that the model predicts the person on the picture to be. You can use the available picture or provide your own. 

## 🤝 Contributing
This project can be used as a base for training your own models on the provided data. Feel free to build upon it and explore the capabilities of CNNs in predicting the age of people from their faces. The Keras model provided can also be further developed. 

## 🙏 Acknowledgments
Special thanks to:
- Le Long Bill
- Anastasiya Minenko
- Alina Minenko
- Danylo Okseniuk 

For their support throughout this project.

## 👤 Owner
**Tychon Kruhlyak**  
📧 Email: [tisha.kruglyak@gmail.com](mailto:tisha.kruglyak@gmail.com)

## 📄 License
This project is licensed under the MIT License. For more information, see the [LICENSE](LICENSE) file.
