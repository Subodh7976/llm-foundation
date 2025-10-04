import yaml
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_split(source_dir: Path, output_file: Path):
    """Reads all .parquet files from a source directory and writes them to a single .jsonl file."""
    logging.info(f"Processing files from: {source_dir}")

    parquet_files = list(source_dir.glob("*.parquet"))
    if not parquet_files:
        logging.warning(f"No .parquet files found in {source_dir}. Skipping.")
        return

    # Ensure the parent directory for the output file exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in tqdm(parquet_files, desc=f"Ingesting {source_dir.name}"):
            try:
                df = pd.read_parquet(file_path)
                if 'text' not in df.columns:
                    logging.warning(
                        f"'text' column not found in {file_path}. Skipping.")
                    continue

                for text in df['text']:
                    # Create a JSON object for each text entry and write it as a new line
                    record = {"text": str(text), "meta": {
                        "source": str(file_path.name)}}
                    f_out.write(json.dumps(record) + '\n')
            except Exception as e:
                logging.error(f"Could not process file {file_path}: {e}")


def main():
    logging.info("--- Starting Step 1: Ingestion ---")

    # Load configuration
    with open("configs/data_sources.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Define paths
    train_dir = Path(config['train_dir'])
    validation_dir = Path(config['validation_dir'])
    output_dir = Path(config['raw_output_dir'])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process both splits
    process_split(train_dir, output_dir / "train.jsonl")
    process_split(validation_dir, output_dir / "validation.jsonl")

    logging.info("--- Finished Step 1: Ingestion ---")


if __name__ == "__main__":
    main()
