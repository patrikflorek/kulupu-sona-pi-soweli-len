# KULUPU SONA PI SOWELI LEN

Using DeepSeek V3 (`deepseek-chat`)

## Steps

1. Download the cleaned Alpaca dataset:

   ```bash
   python scripts/download_alpaca_cleaned.py
   ```

2. Translate the Alpaca dataset to Toki Pona:

   ```bash
   python scripts/translate_en_to_tok.py
   ```

   Note: Translating the English text to Toki Pona gave poor results. We hypothesized that translating the Toki Pona text back to English might help identify inconsistencies and improve the translation.

3. Translate the Alpaca dataset to Toki Pona and back to English:

   ```bash
   python scripts/translate_en_to_tok.py
   ```

   This step translates the instruction, input, and output for both translation directions simultaneously.
