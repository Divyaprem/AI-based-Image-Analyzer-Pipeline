from PIL import Image
import numpy as np
import torchvision.transforms as T

def preprocess_image(img):
    transform = T.Compose([T.ToTensor(), T.Resize((800, 800))])
    return transform(img)

def preprocess_text(text):
    # Example preprocessing: lowercasing and removing extra spaces
    return text.lower().strip()
