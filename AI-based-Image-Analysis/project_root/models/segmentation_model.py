import torch
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os

def load_segmentation_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def segment_and_save_objects(model, img, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_tensor = F.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)
    
    masks = prediction[0]['masks']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    metadata = []
    master_id = np.random.randint(1000, 9999)
    
    for i, mask in enumerate(masks):
        if scores[i] > 0.8:
            binary_mask = mask[0] > 0.8
            object_id = np.random.randint(1000, 9999)
            object_img_path = os.path.join(output_dir, f"object_{object_id}.png")
            
            # Convert mask to 3-channel image
            binary_mask_3ch = binary_mask.repeat(3, 1, 1).float()
            save_image(binary_mask_3ch, object_img_path)
            
            metadata.append({
                "object_id": object_id,
                "master_id": master_id,
                "bbox": prediction[0]['boxes'][i].tolist()
            })
    
    return master_id, metadata
