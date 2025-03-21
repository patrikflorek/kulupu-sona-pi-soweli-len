#!/usr/bin/env python3
"""
Script to translate the Alpaca dataset from English to Toki Pona and back to English
to verify translation quality using OpenAI API directly.
"""
import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Define Pydantic models for the Alpaca dataset
class AlpacaItem(BaseModel):
    """Original Alpaca dataset item."""

    instruction: str
    input: str
    output: str


class AlpacaItemTranslated(BaseModel):
    """Alpaca dataset item with translations to Toki Pona and back to English."""

    instruction: str
    input: str
    output: str
    instruction_tp: str = Field(description="Instruction translated to Toki Pona")
    input_tp: str = Field(description="Input translated to Toki Pona")
    output_tp: str = Field(description="Output translated to Toki Pona")
    instruction_tp_en: str = Field(
        description="Instruction translated back to English from Toki Pona"
    )
    input_tp_en: str = Field(
        description="Input translated back to English from Toki Pona"
    )
    output_tp_en: str = Field(
        description="Output translated back to English from Toki Pona"
    )


# Create the OpenAI client
client = OpenAI(
    api_key="sk-5bb543ef75c749ea9655b33d5edd2225", base_url="https://api.deepseek.com"
)


async def process_item(item: AlpacaItem) -> AlpacaItemTranslated:
    """Process an AlpacaItem: translate to Toki Pona and back to English as a cohesive unit."""
    print(f"Processing: {item.instruction[:50]}...")

    # Prepare a context-aware translation request for the entire item
    context = (
        f"This is a question-answer pair with these parts:\n\n"
        f"INSTRUCTION: {item.instruction}\n\n"
        f"INPUT: {item.input}\n\n"
        f"OUTPUT: {item.output}\n\n"
        f"First, translate each part to Toki Pona, maintaining the relationship between them. "
        f"Then, translate each Toki Pona part back to English. "
        f"Ensure that the back-translations preserve the original meaning as much as possible. "
        f"The instruction is a question or task, the input is additional context (if any), "
        f"and the output is the answer or response to the instruction."
    )

    try:
        # Use the client directly for the complete translation process
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a skilled translator specializing in translating between English and Toki Pona. "
                        "Your task is to translate the given text to Toki Pona and then back to English with these important constraints: "
                        "1. Simplify complex ideas to match Toki Pona's minimalist vocabulary of ~120 words. "
                        "2. Make Toki Pona translations MUCH more concise than the original text - Toki Pona requires brevity. "
                        "3. Focus only on the core meaning and essential information in Toki Pona. "
                        "4. When translating back to English, stay true to what the Toki Pona actually says, without adding information not present in the Toki Pona. "
                        "5. Ensure that all three parts (instruction, input, output) make sense together in both Toki Pona and English. "
                        "6. The back-translation should preserve the original meaning as much as possible given Toki Pona's constraints. "
                        "7. Maintain the relationship between the instruction, input, and output in both languages. "
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
                        f"OUTPUT_TP: [Toki Pona translation of the output]\n"
                        f"INSTRUCTION_TP_EN: [English translation of the Toki Pona instruction]\n"
                        f"INPUT_TP_EN: [English translation of the Toki Pona input]\n"
                        f"OUTPUT_TP_EN: [English translation of the Toki Pona output]"
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
        instruction_tp_en = ""
        input_tp_en = ""
        output_tp_en = ""

        for line in result.split("\n"):
            if line.startswith("INSTRUCTION_TP:"):
                instruction_tp = line.replace("INSTRUCTION_TP:", "").strip()
            elif line.startswith("INPUT_TP:"):
                input_tp = line.replace("INPUT_TP:", "").strip()
            elif line.startswith("OUTPUT_TP:"):
                output_tp = line.replace("OUTPUT_TP:", "").strip()
            elif line.startswith("INSTRUCTION_TP_EN:"):
                instruction_tp_en = line.replace("INSTRUCTION_TP_EN:", "").strip()
            elif line.startswith("INPUT_TP_EN:"):
                input_tp_en = line.replace("INPUT_TP_EN:", "").strip()
            elif line.startswith("OUTPUT_TP_EN:"):
                output_tp_en = line.replace("OUTPUT_TP_EN:", "").strip()

        return AlpacaItemTranslated(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            instruction_tp=instruction_tp,
            input_tp=input_tp,
            output_tp=output_tp,
            instruction_tp_en=instruction_tp_en,
            input_tp_en=input_tp_en,
            output_tp_en=output_tp_en,
        )
    except Exception as e:
        print(f"Error processing item: {e}")
        return AlpacaItemTranslated(
            instruction=item.instruction,
            input=item.input,
            output=item.output,
            instruction_tp="",
            input_tp="",
            output_tp="",
            instruction_tp_en="",
            input_tp_en="",
            output_tp_en="",
        )


async def main():
    """Main function to translate the Alpaca dataset."""
    # Define file paths
    input_file = Path(
        "/home/pato/Code/kulupu_sona_pi_soweli_len/datasets/alpaca_cleaned/alpaca_data_cleaned.json"
    )
    output_dir = Path(
        "/home/pato/Code/kulupu_sona_pi_soweli_len/datasets/soweli_len_wawa"
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Uncomment to test with only a few items
    # data = data[:10]

    # Find the batch to start from by checking existing output files
    existing_batches = sorted(
        [
            int(f.stem.split("_")[-1])
            for f in output_dir.glob("soweli_len_wawa_telo_batch_*.json")
            if f.stem.split("_")[-1].isdigit()
        ]
    )

    start_batch = 0
    if existing_batches:
        start_batch = existing_batches[-1]
        # Check if the last batch is complete (has 100 items or is the final batch)
        last_batch_file = output_dir / f"soweli_len_wawa_telo_batch_{start_batch}.json"
        with open(last_batch_file, "r", encoding="utf-8") as f:
            last_batch_data = json.load(f)

        # If the last batch is complete, start with the next batch
        # Otherwise, we'll reprocess the last batch
        if len(last_batch_data) == 100 or (
            start_batch * 100 + len(last_batch_data) >= len(data)
        ):
            start_batch += 1

    # Calculate starting index
    start_idx = start_batch * 100

    # Check if we've already processed all items
    if start_idx >= len(data):
        print(f"All {len(data)} items have already been processed. Nothing to do.")
        return

    # Process the dataset in batches of 100
    print(
        f"Dataset contains {len(data)} items. Starting from batch {start_batch} (item {start_idx})..."
    )

    batch_size = 100
    total_items = len(data)

    for batch_idx in range(start_batch, (total_items + batch_size - 1) // batch_size):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_items)
        batch_data = data[batch_start:batch_end]

        output_file = output_dir / f"soweli_len_wawa_telo_batch_{batch_idx}.json"
        print(
            f"Processing batch {batch_idx} (items {batch_start}-{batch_end-1}) -> {output_file}"
        )

        translated_data = []

        for i, item in enumerate(batch_data):
            global_idx = batch_start + i
            # Convert dict to Pydantic model
            alpaca_item = AlpacaItem(**item)

            # Process the item (translate to Toki Pona and back to English)
            translated_item = await process_item(alpaca_item)

            # Add to the list
            translated_data.append(
                translated_item.model_dump()
            )  # Using model_dump() instead of dict()

            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == len(batch_data):
                print(
                    f"Processed {global_idx + 1}/{total_items} items ({(global_idx + 1) / total_items * 100:.1f}%)"
                )

        # Save the batch
        print(f"Batch {batch_idx} complete. Saving to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # Create a combined file with all results
    print("Creating combined file with all results...")
    all_translated_data = []
    for batch_idx in range((total_items + batch_size - 1) // batch_size):
        batch_file = output_dir / f"soweli_len_wawa_telo_batch_{batch_idx}.json"
        if batch_file.exists():
            with open(batch_file, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
                all_translated_data.extend(batch_data)

    combined_file = output_dir / "soweli_len_wawa_telo_combined.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_translated_data, f, ensure_ascii=False, indent=2)

    print(f"Done! All translated data saved to {combined_file}")
    print(f"Total items processed: {len(all_translated_data)}/{total_items}")


if __name__ == "__main__":
    asyncio.run(main())
