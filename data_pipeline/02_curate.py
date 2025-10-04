import yaml
import json
from pathlib import Path
from tqdm import tqdm
import logging
import hashlib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def curate_split(input_file: Path, output_file: Path, params: dict, is_training_set: bool):
    """Applies filtering and deduplication to a .jsonl file."""
    logging.info(f"Curating file: {input_file}")

    min_chars = params['min_chars']
    max_chars = params['max_chars']
    perform_dedup = params['deduplication'] and is_training_set

    seen_hashes = set()
    docs_written = 0
    docs_read = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc=f"Curating {input_file.name}"):
            docs_read += 1
            try:
                doc = json.loads(line)
                text = doc['text']

                # 1. Apply length filters
                if not (min_chars <= len(text) <= max_chars):
                    continue

                # 2. Apply deduplication (only for training set)
                if perform_dedup:
                    doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    if doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)

                # If all checks pass, write to output
                f_out.write(line)
                docs_written += 1

            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(
                    f"Skipping malformed line in {input_file}: {e}")

    logging.info(
        f"Finished curating {input_file.name}. Kept {docs_written}/{docs_read} documents.")


def main():
    logging.info("--- Starting Step 2: Curation ---")

    with open("configs/data_sources.yaml", 'r') as f:
        source_config = yaml.safe_load(f)
    with open("configs/processing_params.yaml", 'r') as f:
        params_config = yaml.safe_load(f)

    input_dir = Path(source_config['raw_output_dir'])
    output_dir = Path(source_config['curated_output_dir'])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Curate splits
    curate_split(input_dir / "train.jsonl", output_dir / "train.jsonl",
                 params_config['curation'], is_training_set=True)
    curate_split(input_dir / "validation.jsonl", output_dir /
                 "validation.jsonl", params_config['curation'], is_training_set=False)

    logging.info("--- Finished Step 2: Curation ---")


if __name__ == "__main__":
    main()
