from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model_path = '../model_checkpoint.h5'
model = load_model(model_path)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']

    # Preprocess the image
    img = Image.open(img_file)  # now we just will put the image manually
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class&
    predicted_class = np.argmax(prediction)

    # Get the class labels (assuming you have a list of class labels)
    class_labels = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot','cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika','pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach','sweetcorn','sweetpotato','tomato','turnip','watermelon']
    result = {"class": class_labels[predicted_class], "confidence": float(prediction[0][predicted_class])}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)