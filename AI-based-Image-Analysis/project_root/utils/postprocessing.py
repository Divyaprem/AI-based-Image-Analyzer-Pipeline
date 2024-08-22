def postprocess_identification(predictions, categories):
    """
    Post-process object detection predictions.

    Args:
        predictions (dict): Raw predictions from the detection model.
        categories (list): List of category names.

    Returns:
        dict: Post-processed predictions with category names.
    """
    # Convert tensor predictions to lists if they are tensors
    if hasattr(predictions['boxes'], 'tolist'):
        predictions['boxes'] = predictions['boxes'].tolist()
    if hasattr(predictions['scores'], 'tolist'):
        predictions['scores'] = predictions['scores'].tolist()
    if hasattr(predictions['labels'], 'tolist'):
        predictions['labels'] = predictions['labels'].tolist()
    
    # Ensure labels are integers (index)
    # Convert category indices to category names
    predictions['labels'] = [categories[label] if isinstance(label, int) else label for label in predictions['labels']]
    
    return predictions

def postprocess_segmentation(segmentation_output):
    """
    Post-process segmentation output to prepare for saving or visualization.

    Args:
        segmentation_output (dict): Raw output from the segmentation model.

    Returns:
        dict: Processed segmentation data.
    """
    # Convert tensor outputs to lists if they are tensors
    if hasattr(segmentation_output['masks'], 'cpu'):
        segmentation_output['masks'] = [mask.cpu().numpy() for mask in segmentation_output['masks']]
    if hasattr(segmentation_output['labels'], 'tolist'):
        segmentation_output['labels'] = segmentation_output['labels'].tolist()
    
    return segmentation_output
