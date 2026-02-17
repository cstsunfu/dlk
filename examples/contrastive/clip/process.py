import uuid

from datasets import Dataset
from PIL import Image

from dlk.preprocess import PreProcessor

if __name__ == "__main__":
    # Create simple dummy images to demonstrate multimodal process
    img1 = Image.new("RGB", (224, 224), color="black")
    img2 = Image.new("RGB", (224, 224), color="white")

    data_list = [
        {"image": img1, "text": "A black image.", "uuid": str(uuid.uuid4())},
        {"image": img2, "text": "A white image.", "uuid": str(uuid.uuid4())},
    ] * 50

    dataset = Dataset.from_list(data_list)

    split = dataset.train_test_split(test_size=0.1)
    input_data = {"train": split["train"], "valid": split["test"]}

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
