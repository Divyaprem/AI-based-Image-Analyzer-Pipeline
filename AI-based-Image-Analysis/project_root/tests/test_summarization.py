import unittest
from transformers import pipeline

from models.summarization_model import load_summarization_model, generate_summary


class TestSummarizationModel(unittest.TestCase):

    def test_load_summarization_model(self):
        # Arrange (No specific setup needed)

        # Act
        summarizer = load_summarization_model()

        # Assert
        self.assertIsInstance(summarizer, pipeline)  # Check if it's a Transformers pipeline

    def test_generate_summary(self):
        # Arrange
        texts = ["This is a test sentence for summarization."]

        # Load a pre-trained model (optional)
        summarizer = load_summarization_model()

        # Act
        summaries = generate_summary(summarizer, texts)

        # Assert
        self.assertIsInstance(summaries, list)  # Check for list of summaries
        self.assertEqual(len(texts), len(summaries))  # Ensure matching number of inputs and outputs
        for summary in summaries:
            self.assertIsInstance(summary, str)  # Check if each summary is a string