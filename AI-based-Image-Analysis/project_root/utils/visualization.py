import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def visualize_detection(img_array, boxes, labels):
    """
    Visualize detection results on the image.

    Args:
        img_array (np.array): The image as a NumPy array.
        boxes (list): List of bounding boxes.
        labels (list): List of labels corresponding to the boxes.
    """
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=80)
    ax.imshow(img_array)

    for box, label in zip(boxes, labels):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0], box[1], label, color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.show()

def visualize_segmentation(img_array, masks):
    """
    Visualize segmentation results on the image.

    Args:
        img_array (np.array): The image as a NumPy array.
        masks (list): List of masks where each mask corresponds to a detected object.
    """
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=80)
    ax.imshow(img_array)

    for mask in masks:
        ax.imshow(mask, alpha=0.5, cmap='jet')
    
    plt.axis('off')
    plt.show()

def visualize_summary(mapped_data):
    """
    Visualize the summary data in the form of a table.

    Args:
        mapped_data (dict): Mapped data containing summaries and other attributes.
    """
    fig, ax = plt.subplots()
    table_data = []

    for obj in mapped_data["objects"]:
        table_data.append([
            obj["id"],
            obj["category"],
            obj["description"],
            obj["bounding_box"]
        ])

    table_data.extend([
        [item["text"], item["summary"]]  # Assuming summaries are aligned with text data
        for item in mapped_data["text"]
    ])

    ax.axis('tight')
    ax.axis('off')
    table = plt.table(cellText=table_data,
                      colLabels=["Object ID", "Category", "Description", "Bounding Box", "Text", "Summary"],
                      cellLoc='center',
                      loc='center')

    plt.show()
