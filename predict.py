import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load trained model (inference mode)
model = tf.keras.models.load_model("final_model.keras", compile=False)

# Emotion labels
emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

def prepare_image(img_pil):
    """Preprocess image to match model input (224x224x3)."""
    img = img_pil.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_emotion(image):
    processed_image = prepare_image(image)
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return emotion_labels.get(predicted_class, "Unknown")

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Emotion Detection",
    description="Upload a face image and see the predicted emotion."
)

#  PUBLIC LINK ENABLED
interface.launch(share=True)
