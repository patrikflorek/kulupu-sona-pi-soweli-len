"""
Alpaca Dataset Download Script

This script downloads the cleaned Alpaca dataset from GitHub and saves it locally.
The dataset is used for training conversational AI models and contains instruction-following examples.

The script handles:
- Downloading the dataset from GitHub
- Validating the JSON content
- Saving the dataset to a local directory
- Error handling for various failure scenarios

Usage:
    python download_alpaca_cleaned.py

The dataset will be saved to: datasets/alpaca_cleaned/alpaca_data_cleaned.json
"""

import os
import httpx
import json
from pathlib import Path
from typing import Dict, List, Any


def download_alpaca_dataset() -> None:
    """
    Downloads the cleaned Alpaca dataset from GitHub and saves it to the datasets/alpaca_cleaned directory.

    The function performs the following steps:
    1. Creates the output directory if it doesn't exist
    2. Downloads the dataset from GitHub using HTTPX
    3. Validates the JSON content
    4. Saves the dataset to a local file

    Raises:
        httpx.HTTPStatusError: If the HTTP request fails
        httpx.RequestError: If there's a network-related error
        json.JSONDecodeError: If the downloaded content is not valid JSON
        Exception: For any other unexpected errors
    """
    # URL for the raw content of the JSON file
    # Using raw.githubusercontent.com to get the raw content instead of the GitHub HTML page
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"

    # Define the output directory and file path
    output_dir = Path("datasets/alpaca_cleaned")
    output_file = output_dir / "alpaca_data_cleaned.json"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading Alpaca dataset from {url}")

    try:
        # Send a GET request to the URL using httpx
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the JSON to validate it
            data: List[Dict[str, Any]] = response.json()

            # Save the JSON data to the output file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Successfully downloaded and saved to {output_file}")
            print(f"Dataset contains {len(data)} entries")

    except httpx.HTTPStatusError as e:
        print(f"Error downloading the dataset: {e}")
    except httpx.RequestError as e:
        print(f"Error with the request: {e}")
    except json.JSONDecodeError:
        print("Error: The downloaded content is not valid JSON")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    download_alpaca_dataset()
