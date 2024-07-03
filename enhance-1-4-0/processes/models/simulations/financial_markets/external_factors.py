import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_external_factors(api_url, start_date, end_date):
    """
    Fetches external factor data from a specified API between the given date range.

    Args:
        api_url (str): The URL of the external factor data API.
        start_date (str): The start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): The end date for data retrieval in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary with dates as keys and external factors as values.
    """
    try:
        logger.info(f"Fetching external factors from {api_url} for date range {start_date} to {end_date}")
        
        # Example payload for API request
        payload = {
            'start_date': start_date,
            'end_date': end_date
        }

        # Send request to the API
        response = requests.get(api_url, params=payload)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Convert the data to a dictionary with dates as keys
        external_factors = {entry['date']: entry['value'] for entry in data}

        logger.info("External factors fetched successfully")
        return external_factors

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching external factors: {str(e)}", exc_info=True)
        return {}