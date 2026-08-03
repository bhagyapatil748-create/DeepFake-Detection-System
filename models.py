import torch
import torch.nn as nn
import timm

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# IMAGE MODEL
# ==========================================================

def create_image_model():
    """
    Creates the EfficientNet-B0 model used for image detection.
    """

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False
    )

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )

    return model


# ==========================================================
# VIDEO MODEL
# ==========================================================

class DeepFakeVideoModel(nn.Module):

    def __init__(self):
        super().__init__()

        # EfficientNet Backbone
        self.feature_extractor = timm.create_model(
            "efficientnet_b0",
            pretrained=False
        )

        # Remove classification layer
        self.feature_extractor.classifier = nn.Identity()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(512, 2)

    def forward(self, x):

        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size * seq_len, c, h, w)

        features = self.feature_extractor(x)

        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)

        output = lstm_out[:, -1, :]

        output = self.dropout(output)

        output = self.fc(output)

        return output


def create_video_model():
    """
    Creates the DeepFake Video Model.
    """
    return DeepFakeVideoModel()


# ==========================================================
# LOAD IMAGE MODEL
# ==========================================================

def load_image_model(model_path):

    model = create_image_model()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


# ==========================================================
# LOAD VIDEO MODEL
# ==========================================================

def load_video_model(model_path):

    model = create_video_model()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


# ==========================================================
# TEST MODE
# ==========================================================

if __name__ == "__main__":

    IMAGE_MODEL = "best_image_model.pth"
    VIDEO_MODEL = "best_video_model.pth"

    print("=" * 50)
    print("Loading Image Model...")
    print("=" * 50)

    image_model = load_image_model(IMAGE_MODEL)

    print("✅ Image Model Loaded")

    print()

    print("=" * 50)
    print("Loading Video Model...")
    print("=" * 50)

    video_model = load_video_model(VIDEO_MODEL)

    print("✅ Video Model Loaded")