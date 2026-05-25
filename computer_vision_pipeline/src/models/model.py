import timm
import torch


CLASS_NAMES = ["heart", "oblong", "oval", "round", "square"]
NUM_CLASSES = len(CLASS_NAMES)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(model_name: str = "inception_v4", pretrained: bool = True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    return model