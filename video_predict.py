import torch
import torch.nn
import os
import time

from models import load_video_model
from utils import extract_frames
from utils import get_video_info
import torch.nn.functional as F

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = "best_video_model.pth"

video_model = load_video_model(MODEL_PATH)

# ==========================================================
# CLASS LABELS
# ==========================================================

CLASS_NAMES = {
    0: "Fake",
    1: "Real"
}

# ==========================================================
# VIDEO PREDICTION
# ==========================================================

def predict_video(video_path):

    start_time = time.time()

    # ---------------------------------------------
    # Extract Frames
    # ---------------------------------------------

    frames = extract_frames(video_path)

    if len(frames) == 0:
        raise Exception("No frames extracted from video.")

    # ---------------------------------------------
    # Prepare Input
    # ---------------------------------------------

    frames = frames.unsqueeze(0).to(device)

    # Shape becomes:
    #
    # (1,20,3,224,224)

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    with torch.no_grad():

        outputs = video_model(frames)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    prediction = CLASS_NAMES[predicted.item()]

    confidence = confidence.item() * 100

    probability_dict = {
        "Fake": round(probabilities[0][0].item() * 100, 2),
        "Real": round(probabilities[0][1].item() * 100, 2)
    }

    processing_time = round(time.time() - start_time, 3)

    # ---------------------------------------------
    # Video Info
    # ---------------------------------------------

    info = get_video_info(video_path)

    return (
        prediction,
        round(confidence,2),
        probability_dict,
        processing_time,
        info
    )

# ==========================================================
# TEST MODE
# ==========================================================

if __name__ == "__main__":

    VIDEO_PATH = "sample_media/test.mp4"

    prediction, confidence, probabilities, processing_time, info = predict_video(VIDEO_PATH)

    print("="*60)

    print("VIDEO PREDICTION")

    print("="*60)

    print()

    print("Prediction :", prediction)

    print()

    print("Confidence :", confidence,"%")

    print()

    print("Probabilities")

    print(probabilities)

    print()

    print("Processing Time :", processing_time,"seconds")

    print()

    print("Video Information")

    print(info)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_video_model.pth"
)
