"""
Toki Pona Translation Script

This script translates the Alpaca dataset from English to Toki Pona using the OpenAI API.
The script handles:
- Loading the dataset from a JSON file
- Translating each item's instruction, input, and output fields
- Saving the translated dataset to a new JSON file

Usage:
    python translate_en_to_tok.py

The translated dataset will be saved to: datasets/soweli_len/soweli_len_telo.json
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with environment variables
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


class AlpacaItem(BaseModel):
    """Original Alpaca dataset item."""

    instruction: str
    input: str
    output: str


class AlpacaItemTranslated(BaseModel):
    """Alpaca dataset item with Toki Pona translations."""

    instruction: str
    input: str
    output: str
    instruction_tp: str = Field(description="Instruction translated to Toki Pona")
    input_tp: str = Field(description="Input translated to Toki Pona")
    output_tp: str = Field(description="Output translated to Toki Pona")


async def translate_text(text: str) -> str:
    """Translate a single text from English to Toki Pona."""
    if not text.strip():
        return ""

    try:
        # Use the client directly instead of the agent
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a skilled translator specializing in translating from English to Toki Pona. "
                        "Your task is to translate the given text to Toki Pona with these important constraints: "
                        "1. Simplify complex ideas to match Toki Pona's minimalist vocabulary of ~120 words. "
                        "2. Make translations MUCH more concise than the original text - Toki Pona requires brevity. "
                        "3. Focus only on the core meaning and essential information. "
                        "4. Omit details, examples, and explanations that aren't absolutely necessary. "
                        "5. Use Toki Pona's grammar structures appropriately. "
                        "6. Keep translations short and simple - under 100 words whenever possible. "
                        "7. Remember that Toki Pona is designed for simplicity - complex ideas must be reduced to basic concepts. "
                        "Provide only the translated text without explanations or additional comments."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Translate this English text to Toki Pona, making it much simpler and more concise: {text}",
                },
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating text: {e}")
        return ""


async def translate_item(item: AlpacaItem) -> AlpacaItemTranslated:
    """Translate an Alpaca dataset item to Toki Pona.

    Args:
        item: The Alpaca dataset item to translate

    Returns:
        The translated Alpaca item

    Raises:
        Exception: If any error occurs during translation
    """
    print(f"Translating: {item.instruction[:50]}...")

    # Prepare a context-aware translation request
    context = (
        f"This is a question-answer pair with these parts:\n\n"
        f"INSTRUCTION: {item.instruction}\n\n"
        f"INPUT: {item.input}\n\n"
        f"OUTPUT: {item.output}\n\n"
        f"Translate each part to Toki Pona separately, maintaining the relationship between them. "
        f"The instruction is a question or task, the input is additional context (if any), "
        f"and the output is the answer or response to the instruction."
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a skilled translator specializing in translating from English to Toki Pona. "
                        "Your task is to translate the given text to Toki Pona with these important constraints: "
                        "1. Simplify complex ideas to match Toki Pona's minimalist vocabulary of ~120 words. "
                        "2. Make translations MUCH more concise than the original text - Toki Pona requires brevity. "
                        "3. Focus only on the core meaning and essential information. "
                        "4. Omit details, examples, and explanations that aren't absolutely necessary. "
                        "5. Use Toki Pona's grammar structures appropriately. "
                        "6. Keep translations short and simple - under 100 words whenever possible. "
                        "7. Remember that Toki Pona is designed for simplicity - complex ideas must be reduced to basic concepts. "
                        "8. Understand the relationship between instruction, input, and output parts. "
                        "Provide only the translated text without explanations or additional comments."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        f"Format your response exactly like this:\n"
                        f"INSTRUCTION_TP: [Toki Pona translation of the instruction]\n"
                        f"INPUT_TP: [Toki Pona translation of the input]\n"
                        f"OUTPUT_TP: [Toki Pona translation of the output]"
                    ),
                },
            ],
            stream=False,
        )

        result = response.choices[0].message.content.strip()

        # Parse the response to extract the translations
        instruction_tp = ""
        input_tp = ""
        output_tp = ""

        for line in result.split("\n"):
            if line.startswith("INSTRUCTION_TP:"):
                instruction_tp = line.replace("INSTRUCTION_TP:", "").strip()
            elif line.startswith("INPUT_TP:"):
                input_tp = line.replace("INPUT_TP:", "").strip()
            elif line.startswith("OUTPUT_TP:"):
                output_tp = line.replace("OUTPUT_TP:", "").strip()

        return AlpacaItemTranslated(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            instruction_tp=instruction_tp,
            input_tp=input_tp,
            output_tp=output_tp,
        )
    except Exception as e:
        print(f"Error translating item: {e}")
        raise


async def main() -> None:
    """Main function to translate the Alpaca dataset."""
    input_file = Path("datasets/alpaca_cleaned/alpaca_data_cleaned.json")
    output_dir = Path("datasets/soweli_len")
    output_file = output_dir / "soweli_len_telo.json"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # Test with only three items
    data = data[:3]

    # Process the dataset
    print(f"Dataset contains {len(data)} items. Starting translation...")
    translated_data: List[Dict[str, Any]] = []

    for i, item in enumerate(data):
        alpaca_item = AlpacaItem(**item)
        translated_item = await translate_item(alpaca_item)
        translated_data.append(translated_item.dict())

        if (i + 1) % 10 == 0:
            print(f"Translated {i + 1}/{len(data)} items")

    # Save the translated dataset
    print(f"Saving translated dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    print("Translation complete!")


if __name__ == "__main__":
    asyncio.run(main())
