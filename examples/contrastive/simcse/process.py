import uuid

from datasets import load_dataset

from dlk.preprocess import PreProcessor


def preprocess_function(batch):
    return {
        "sentence_a": batch["text"],
        "sentence_b": batch[
            "text"
        ],  # SimCSE trick: use identical input text, variance is created by dropout
        "uuid": [str(uuid.uuid4()) for _ in range(len(batch["text"]))],
    }


if __name__ == "__main__":
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(
        range(1000)
    )
    data = data.filter(lambda x: len(x["text"].strip()) > 10)
    data = data.map(preprocess_function, batched=True, remove_columns=["text"])

    split = data.train_test_split(test_size=0.1)
    input_data = {"train": split["train"], "valid": split["test"]}

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
