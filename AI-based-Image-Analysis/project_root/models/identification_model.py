import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as F
from PIL import Image

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model

def make_prediction(model, img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction['labels'] = [categories[label] for label in prediction['labels']]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = F.to_tensor(img)  # Convert to tensor
    img_tensor = img_tensor * 255.0  # Scale back to [0-255] for visualization
    img_tensor = img_tensor.byte()  # Convert to byte tensor

    # Draw bounding boxes on the image
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"], colors="red", width=2)

    # Convert to NumPy array for display
    img_with_bboxes_np = img_with_bboxes.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    return img_with_bboxes_np
