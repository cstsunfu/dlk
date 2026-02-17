import os
import uuid

from datasets import Dataset

from dlk.preprocess import PreProcessor


def generate_dummy_data(num_samples=100):
    """Generate dummy classification data."""
    sentences = [
        "This is a positive sentence." if i % 2 == 0 else "This is a negative sentence."
        for i in range(num_samples)
    ]
    labels = ["pos" if i % 2 == 0 else "neg" for i in range(num_samples)]
    uuids = [str(uuid.uuid4()) for _ in range(num_samples)]
    return Dataset.from_dict(
        {"sentence": sentences, "labels": [[l] for l in labels], "uuid": uuids}
    )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # 1. Generate Dataset
    dummy_dataset = generate_dummy_data()
    input_data = {"train": dummy_dataset, "valid": dummy_dataset.select(range(10))}

    # 2. Run Processor
    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
    print("Preprocessing finished. Data saved to ./data/processed_data")
