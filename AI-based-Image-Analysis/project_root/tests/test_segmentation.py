import unittest
import torch

from models.segmentation_model import load_segmentation_model, segment_and_save_objects


class TestSegmentationModel(unittest.TestCase):

    def test_load_segmentation_model(self):
        # Arrange (No specific setup needed)

        # Act
        model = load_segmentation_model()

        # Assert
        self.assertIsInstance(model, torch.nn.Module)  # Check if it's a PyTorch model

    def test_segment_and_save_objects(self):
        # Arrange
        # Replace with a dummy image for testing
        dummy_img = torch.randn(3, 224, 224)

        # Load a pre-trained model (optional)
        model = load_segmentation_model()

        # Act (Skip actual saving for testing)
        master_id, metadata = segment_and_save_objects(model, dummy_img, None)

        # Assert
        self.assertIsInstance(master_id, int)  # Check for generated master ID
        self.assertIsInstance(metadata, list)  # Check for list of detected objects