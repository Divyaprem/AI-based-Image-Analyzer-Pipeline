import unittest
import torch

from models.identification_model import load_model, make_prediction


class TestIdentificationModel(unittest.TestCase):

    def test_load_model(self):
        # Arrange (No specific setup needed)

        # Act
        model = load_model()

        # Assert
        self.assertIsInstance(model, torch.nn.Module)  # Check if it's a PyTorch model

    def test_make_prediction(self):
        # Arrange
        # Replace with a dummy image for testing
        dummy_img = torch.randn(3, 224, 224)

        # Load a pre-trained model (optional)
        model = load_model()

        # Act
        prediction = make_prediction(model, dummy_img)

        # Assert
        self.assertIn('boxes', prediction.keys())  # Check for presence of bounding boxes
        self.assertIn('scores', prediction.keys())  # Check for presence of confidence scores
        self.assertIn('labels', prediction.keys())  # Check for presence of predicted labels