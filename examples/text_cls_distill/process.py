# process.py
import uuid

from datasets import Dataset

from dlk.preprocess import PreProcessor


def add_uuid(batch):
    batch["uuid"] = [str(uuid.uuid4()) for _ in range(len(batch["sentence"]))]
    return batch


if __name__ == "__main__":
    raw_data = [
        {"sentence": "This is a good movie.", "labels": ["pos"]},
        {"sentence": "A very bad experience.", "labels": ["neg"]},
        {"sentence": "It is just okay.", "labels": ["neg"]},
        {"sentence": "Great job everyone!", "labels": ["pos"]},
    ] * 25  # 扩充到100条数据

    dataset = Dataset.from_list(raw_data).map(add_uuid, batched=True)

    split = dataset.train_test_split(test_size=0.2)
    input_data = {
        "train": split["train"],
        "valid": split["test"],
    }

    # 2. 运行 PreProcessor（此时会调用 HTTP API 获取 Teacher Logits）
    print("Starting PreProcessing and fetching Teacher Labels...")
    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
    print("PreProcessing Finished! Data is saved to ./data/processed_data")
