import uuid

import pandas as pd

from dlk.preprocess import PreProcessor

data = []
data.append(
    {
        "input": """23 October 2015 Last updated at 17:44 BST It's the highest rating a tropical storm can get and is the first one of this magnitude to hit mainland Mexico since 1959. But how are the categories decided and what do they mean? Newsround reporter Jenny Lawrence explains.""",
        "target": """Hurricane Patricia has been rated as a category 5 storm.""",
        "uuid": str(uuid.uuid1()),
    }
)
data.append(
    {
        "input": """Christopher Williams, 25, who was living in Derby, died at the scene of the crash on the A52, in Bottesford, on 25 May 2016. Garry Allen, 33, of Cressing Road, Braintree, Essex, was arrested at the time and has now been charged with causing death by dangerous driving. He is due to appear at Leicester Magistrates' Court on Friday.""",
        "target": """A man has been charged nearly a year after a collision in which a motorcyclist died in Leicestershire.""",
        "uuid": str(uuid.uuid1()),
    }
)
# data = [data[0], data[0]]
input = {
    "train": pd.DataFrame(data).head(100),
    "valid": pd.DataFrame(data).head(100),
}

processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
