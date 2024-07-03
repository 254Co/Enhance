import openai
import logging
from config.openai_config import OPENAI_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIIntegration:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def generate_text(self, prompt, model="text-davinci-003", max_tokens=100, temperature=0.7):
        try:
            logger.info(f"Generating text with model: {model}, prompt: {prompt}")
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None
            )
            generated_text = response.choices[0].text.strip()
            logger.info(f"Generated text: {generated_text}")
            return generated_text
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}", exc_info=True)
            raise

    def classify_text(self, text, labels, model="text-davinci-003"):
        try:
            logger.info(f"Classifying text with model: {model}, text: {text}, labels: {labels}")
            response = openai.Classification.create(
                model=model,
                query=text,
                labels=labels
            )
            classification_label = response.label
            logger.info(f"Classified label: {classification_label}")
            return classification_label
        except Exception as e:
            logger.error(f"Error in classify_text: {str(e)}", exc_info=True)
            raise

    def summarize_text(self, text, model="text-davinci-003", max_tokens=100):
        try:
            logger.info(f"Summarizing text with model: {model}, text: {text}")
            prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
            summary = self.generate_text(prompt, model=model, max_tokens=max_tokens)
            logger.info(f"Summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error in summarize_text: {str(e)}", exc_info=True)
            raise