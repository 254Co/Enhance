import pytest
from processes.LLMs.integration import OpenAIIntegration
from unittest.mock import patch

@pytest.fixture(scope="module")
def openai_integration():
    return OpenAIIntegration()

def test_generate_text(openai_integration):
    with patch('openai.Completion.create') as mock_create:
        mock_create.return_value.choices = [type('obj', (object,), {'text': 'Generated text'})]
        result = openai_integration.generate_text("Hello")
        assert result == "Generated text"

def test_classify_text(openai_integration):
    with patch('openai.Classification.create') as mock_create:
        mock_create.return_value.label = 'Test label'
        result = openai_integration.classify_text("Test text", ["Label1", "Label2"])
        assert result == "Test label"

def test_summarize_text(openai_integration):
    with patch('openai.Completion.create') as mock_create:
        mock_create.return_value.choices = [type('obj', (object,), {'text': 'Summary text'})]
        result = openai_integration.summarize_text("Long text")
        assert result == "Summary text"