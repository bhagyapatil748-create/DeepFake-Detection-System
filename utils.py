import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import os

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# IMAGE SIZE
# ==========================================================

IMAGE_SIZE = 224


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image):

    """
    PIL Image -> Tensor
    """

    image = image.convert("RGB")

    tensor = image_transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


# ==========================================================
# EXTRACT VIDEO FRAMES
# ==========================================================

def extract_frames(video_path, num_frames=20):

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        cap.release()
        return []

    indices = np.linspace(
        0,
        frame_count - 1,
        num_frames,
        dtype=int
    )

    frames = []

    current = 0

    selected = set(indices)

    while True:

        success, frame = cap.read()

        if not success:
            break

        if current in selected:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(frame)

            frame = image_transform(frame)

            frames.append(frame)

        current += 1

    cap.release()

    return torch.stack(frames)


# ==========================================================
# VIDEO INFORMATION
# ==========================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        "fps": round(fps, 2),
        "frames": frame_count,
        "duration": round(duration, 2),
        "width": width,
        "height": height
    }


# ==========================================================
# FILE TYPE
# ==========================================================

def get_file_type(filename):

    ext = os.path.splitext(filename)[1].lower()

    image_ext = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]

    video_ext = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv"
    ]

    if ext in image_ext:
        return "image"

    if ext in video_ext:
        return "video"

    return "unknown"


# ==========================================================
# TIMER
# ==========================================================

def start_timer():
    return time.time()


def stop_timer(start_time):

    return round(time.time() - start_time, 3)