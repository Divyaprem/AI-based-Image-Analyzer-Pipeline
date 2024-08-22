import unittest
from easyocr import Reader

from models.text_extraction_model import load_easyocr_model, extract_text


class TestTextExtraction(unittest.TestCase):

    def test_load_easyocr_model(self):
        # Arrange (No specific setup needed)

        # Act
        reader = load_easyocr_model()

        # Assert
        self.assertIsInstance(reader, Reader)  # Check if it's an EasyOCR reader

    def test_extract_text(self):
        # Arrange
        # Replace with a dummy image for testing (PIL Image format)
        dummy_img = ...

        # Load a pre-trained model (optional)
        reader = load_easyocr_model()

        # Act
        extracted_text = extract_text(reader, dummy_img)

        # Assert
        self.assertIsInstance(extracted_text, list)  # Check for list of extracted text