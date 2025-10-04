import yaml
import json
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def format_split(input_file: Path, output_file: Path, tokenizer: Tokenizer, context_length: int):
    """
    Tokenizes a .jsonl file and packs it into a memory-mapped numpy array using a
    memory-efficient two-pass strategy.
    """
    logging.info(
        f"Formatting {input_file.name} with context length {context_length}.")

    # Get EOS token ID to append after each document
    eos_token_id = tokenizer.token_to_id("[EOS]")
    if eos_token_id is None:
        raise ValueError("[EOS] token not found in tokenizer vocabulary!")

    # --- Pass 1: Count total tokens to pre-allocate memory ---
    logging.info("Pass 1/2: Counting total tokens...")
    total_tokens = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Counting tokens in {input_file.name}"):
            try:
                text = json.loads(line)['text']
                # Add 1 for the EOS token that will be appended
                total_tokens += len(tokenizer.encode(text).ids) + 1
            except (json.JSONDecodeError, KeyError):
                continue

    logging.info(f"Found {total_tokens:,} total tokens in {input_file.name}.")

    # --- Prepare for Pass 2 ---
    num_samples = total_tokens // context_length
    if num_samples == 0:
        logging.warning(
            f"Not enough tokens to create a single sample for {input_file.name}. Skipping.")
        return

    # Create a memory-mapped array. This is an array on disk that numpy treats like an array in memory.
    # This is the key to not using RAM.
    output_path = output_file.with_suffix('.npy')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # We use uint16 since vocab size < 65,535, saving 50% memory over default int32/64
    arr = np.memmap(output_path, dtype=np.uint16, mode='w+',
                    shape=(num_samples * context_length,))

    # --- Pass 2: Tokenize and fill the memory-mapped array ---
    logging.info("Pass 2/2: Tokenizing and writing to disk...")
    token_idx = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Tokenizing and filling {input_file.name}"):
            try:
                text = json.loads(line)['text']
                token_ids = tokenizer.encode(text).ids
                token_ids.append(eos_token_id)

                # Write the tokens to the memory-mapped array
                if token_idx + len(token_ids) <= len(arr):
                    arr[token_idx: token_idx + len(token_ids)] = token_ids
                    token_idx += len(token_ids)
            except (json.JSONDecodeError, KeyError):
                continue

    # Reshape the 1D array on disk to our desired (num_samples, context_length) shape
    # The flush ensures all changes are written to the file.
    arr.reshape((num_samples, context_length))
    arr.flush()

    logging.info(f"Saved {num_samples:,} samples to {output_path}")


def main():
    logging.info("--- Starting Step 4: Formatting for Pretraining ---")

    with open("configs/data_sources.yaml", 'r') as f:
        source_config = yaml.safe_load(f)
    with open("configs/processing_params.yaml", 'r') as f:
        params_config = yaml.safe_load(f)

    curated_dir = Path(source_config['curated_output_dir'])
    tokenizer_path = Path(
        source_config['tokenizer_output_dir']) / "bpe_tokenizer.json"
    output_dir = Path(source_config['tokenized_output_dir'])
    context_length = params_config['formatting']['context_length']

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at {tokenizer_path}. Please run step 3 first.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    format_split(curated_dir / "train.jsonl", output_dir /
                 "train", tokenizer, context_length)
    format_split(curated_dir / "validation.jsonl", output_dir /
                 "validation", tokenizer, context_length)

    logging.info("--- Finished Step 4: Formatting for Pretraining ---")


if __name__ == "__main__":
    main()
