import uuid

from datasets import Dataset

from dlk.preprocess import PreProcessor

if __name__ == "__main__":
    data = []
    data.append(
        {
            "input": """23 October 2015 Last updated at 17:44 BST It's the highest rating a tropical storm can get and is the first one of this magnitude to hit mainland Mexico since 1959. But how are the categories decided and what do they mean? Newsround reporter Jenny Lawrence explains.""",
            "target": """Hurricane Patricia has been rated as a category 5 storm.""",
            "uuid": str(uuid.uuid4()),
        }
    )
    data.append(
        {
            "input": """Christopher Williams, 25, who was living in Derby, died at the scene of the crash on the A52, in Bottesford, on 25 May 2016. Garry Allen, 33, of Cressing Road, Braintree, Essex, was arrested at the time and has now been charged with causing death by dangerous driving. He is due to appear at Leicester Magistrates' Court on Friday.""",
            "target": """A man has been charged nearly a year after a collision in which a motorcyclist died in Leicestershire.""",
            "uuid": str(uuid.uuid4()),
        }
    )

    # Convert to HF Dataset
    dataset = Dataset.from_list(data)

    input_data = {
        "train": dataset,
        "valid": dataset,  # Reuse for demo
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
