import torch
import torch.nn.functional as F
from PIL import Image
import time
import os

from models import load_image_model
from utils import preprocess_image

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LOAD MODEL (ONLY ONCE)
# ==========================================================

MODEL_PATH = "best_image_model.pth"

image_model = load_image_model(MODEL_PATH)

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = {
    0: "Fake",
    1: "Real"
}

# ==========================================================
# IMAGE PREDICTION
# ==========================================================

def predict_image(image_path):

    """
    Predict whether an image is Real or Fake.

    Returns:
        prediction (str)
        confidence (float)
        probabilities (dict)
        processing_time (float)
    """

    start_time = time.time()

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    tensor = preprocess_image(image)

    # Prediction
    with torch.no_grad():

        outputs = image_model(tensor)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    prediction = CLASS_NAMES[predicted.item()]

    confidence = confidence.item() * 100

    processing_time = time.time() - start_time

    probability_dict = {
        "Fake": round(probabilities[0][0].item() * 100, 2),
        "Real": round(probabilities[0][1].item() * 100, 2)
    }

    return (
        prediction,
        round(confidence, 2),
        probability_dict,
        round(processing_time, 3)
    )


# ==========================================================
# TEST MODE
# ==========================================================

if __name__ == "__main__":

    IMAGE_PATH = "sample_media/test.jpg"

    prediction, confidence, probabilities, processing_time = predict_image(IMAGE_PATH)

    print("=" * 50)
    print("Prediction")
    print("=" * 50)

    print("Result :", prediction)

    print("Confidence :", confidence)

    print()

    print("Probabilities")

    print(probabilities)

    print()

    print("Processing Time :", processing_time, "seconds")
    import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_image_model.pth"
)