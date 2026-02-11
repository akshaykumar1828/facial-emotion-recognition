import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# Load TFLite Model
# ===============================

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotion labels
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ===============================
# Image Preprocessing
# ===============================

def prepare_image(image):
    # Ensure 3 channels
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(image)

    # Match model dtype
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(input_details[0]['dtype'])

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ===============================
# Prediction Function
# ===============================

def predict_emotion(image):
    processed_image = prepare_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(prediction)

    return emotion_labels[predicted_class]


# ===============================
# Gradio Interface
# ===============================

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Emotion Detection",
    description="Upload a face image and see the predicted emotion."
)

# For HuggingFace Spaces
interface.launch(server_name="0.0.0.0", server_port=7860)
