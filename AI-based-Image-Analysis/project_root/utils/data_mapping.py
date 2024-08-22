import json

def map_data_to_objects(metadata, processed_text, summaries, descriptions):
    """
    Map metadata to objects, including extracted text and summaries.

    Args:
        metadata (list): List of dictionaries containing object metadata.
        processed_text (list): List of processed text data extracted from the image.
        summaries (list): List of summaries for each text entry.
        descriptions (list): List of descriptions for each detected category.

    Returns:
        dict: A dictionary with mapped data.
    """
    mapped_data = {"objects": [], "text": []}

    # Process object metadata
    for obj in metadata:
        obj_id = obj.get('id', 'N/A')
        obj_category = obj.get('category', 'N/A')
        obj_description = descriptions[obj.get('category_id', 0)] if obj.get('category_id', 0) < len(descriptions) else 'N/A'
        bounding_box = obj.get('bounding_box', 'N/A')

        mapped_data["objects"].append({
            "id": obj_id,
            "category": obj_category,
            "description": obj_description,
            "bounding_box": bounding_box
        })

    # Process text and summaries
    for text, summary in zip(processed_text, summaries):
        mapped_data["text"].append({
            "text": text,
            "summary": summary
        })

    return mapped_data

def save_data_mapping(mapped_data, output_path):
    """
    Save the mapped data to a JSON file.

    Args:
        mapped_data (dict): Mapped data to save.
        output_path (str): Path to save the JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(mapped_data, f, indent=4)
