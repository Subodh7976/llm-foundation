# data_pipeline/03_tokenize.py
import yaml
import json
from pathlib import Path
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_training_corpus(file_path: Path, sample_size: int):
    # ... (this function remains unchanged)
    logging.info(
        f"Reading corpus from {file_path} with sample size {sample_size} bytes.")
    bytes_read = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if bytes_read >= sample_size:
                break
            bytes_read += len(line.encode('utf-8'))
            try:
                yield json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                continue


def main():
    logging.info("--- Starting Step 3: Tokenizer Training ---")

    with open("configs/data_sources.yaml", 'r') as f:
        source_config = yaml.safe_load(f)
    with open("configs/processing_params.yaml", 'r') as f:
        params_config = yaml.safe_load(f)

    curated_data_dir = Path(source_config['curated_output_dir'])
    train_file = curated_data_dir / "train.jsonl"
    output_dir = Path(source_config['tokenizer_output_dir'])
    tokenizer_params = params_config['tokenizer']

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- MODIFICATION: Read special tokens from config ---
    st_config = tokenizer_params['special_tokens']
    unk_token = st_config['unk_token']

    # Combine all special tokens for the trainer
    all_special_tokens = [st_config['unk_token'], st_config['pad_token'],
                          st_config['bos_token'], st_config['eos_token']]
    all_special_tokens.extend(st_config.get('additional_special_tokens', []))

    # 1. Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token=str(unk_token))
                          )  # Ensure it's a string
    tokenizer.pre_tokenizer = Whitespace()

    # 2. Initialize a trainer
    trainer = BpeTrainer(
        vocab_size=tokenizer_params['vocab_size'],
        special_tokens=all_special_tokens
    )

    # 3. Train the tokenizer
    logging.info(
        f"Training tokenizer with vocab size {tokenizer_params['vocab_size']}...")
    logging.info(f"Using special tokens: {all_special_tokens}")
    corpus_iterator = get_training_corpus(
        train_file, tokenizer_params['training_sample_size'])
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    # --- Set post-processor and padding for HF compatibility ---
    # This ensures that when we use this tokenizer with Hugging Face later,
    # it knows how to handle things like padding and templates correctly.
    tokenizer.add_special_tokens([st_config['pad_token']])

    # 4. Save the tokenizer
    output_path = output_dir / "bpe_tokenizer.json"
    tokenizer.save(str(output_path))

    logging.info(f"Tokenizer trained and saved to: {output_path}")
    logging.info("--- Finished Step 3: Tokenizer Training ---")


if __name__ == "__main__":
    main()
