import torch
import torchvision
import timm
import pandas
import numpy
import matplotlib
import sklearn
import cv2
from PIL import Image

from computer_vision_pipeline.src.utils.paths import PROJECT_ROOT, DATA_DIR
from computer_vision_pipeline.src.data.transforms import get_train_transforms, get_eval_transforms
from computer_vision_pipeline.src.models.model import create_model, get_device, CLASS_NAMES

print("All imports work.")
print("Project root:", PROJECT_ROOT)
print("Data dir:", DATA_DIR)
print("Classes:", CLASS_NAMES)
print("Device:", get_device())

model = create_model(pretrained=False)
print("Model created successfully:", type(model).__name__)