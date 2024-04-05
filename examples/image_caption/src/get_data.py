import os
import pickle as pkl
import uuid

import requests
from PIL import Image
from tqdm import tqdm

# From phiyodr/coco2017
_data_pair = [
    (
        "http://images.cocodataset.org/train2017/000000391895.jpg",
        "A man with a red helmet on a small moped on a dirt road.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000391895.jpg",
        "Man riding a motor bike on a dirt road on the countryside.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000391895.jpg",
        "A man riding on the back of a motorcycle.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000391895.jpg",
        "A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000391895.jpg",
        "A man in a red shirt and a red hat is on a motorcycle on a hill side.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "A woman wearing a net on her head cutting a cake.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "A woman cutting a large white sheet cake.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "A woman wearing a hair net cutting a large sheet cake.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "there is a woman that is cutting a white cake",
    ),
    (
        "http://images.cocodataset.org/train2017/000000522418.jpg",
        "A woman marking a cake with the back of a chef's knife.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "A child holding a flowered umbrella and petting a yak.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "A young man holding an umbrella next to a herd of cattle.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "a young boy barefoot holding an umbrella touching the horn of a cow",
    ),
    (
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "A young boy with an umbrella who is touching the horn of a cow.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000184613.jpg",
        "A boy holding an umbrella while standing next to livestock.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "A young boy standing in front of a computer keyboard.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "a little boy wearing headphones and looking at a computer monitor",
    ),
    (
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "He is listening intently to the computer at school.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "A young boy stares up at the computer monitor.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000318219.jpg",
        "a young kid with head phones on using a computer",
    ),
    (
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "a boy wearing headphones using one computer in a long row of computers",
    ),
    (
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "A little boy with earphones on listening to something.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "A group of people sitting at desk using computers.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "Children sitting at computer stations on a long table.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000554625.jpg",
        "A small child wearing headphones plays on the computer.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "A woman in a room with a cat.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "A girl smiles as she holds a cat and wears a brightly colored skirt.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "a woman is holding a cat in her kitchen",
    ),
    (
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "A woman is working in a kitchen carrying a soft toy.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000574769.jpg",
        "A woman is holding a cat in her kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "A young girl inhales with the intent of blowing out a candle.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "A young girl is preparing to blow out her candle.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "A kid is to blow out the single candle in a bowl of birthday goodness.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "Girl blowing out the candle on an ice-cream",
    ),
    (
        "http://images.cocodataset.org/train2017/000000060623.jpg",
        "A little girl is getting ready to blow out a candle on a small dessert.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "A commercial stainless kitchen with a pot of food cooking.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "Some food sits in a pot in a kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "A kitchen has all stainless steel appliances and counters.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "a kitchen with a sink and many cooking machines and a pot of food",
    ),
    (
        "http://images.cocodataset.org/train2017/000000309022.jpg",
        "Food cooks in a pot on a stove in a kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "Two men wearing aprons working in a commercial-style kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "Chefs preparing food in a professional metallic style kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "Two people standing around in a large kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "A commercial kitchen with two men working to prepare several plates.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000005802.jpg",
        "two men in white shirts in a large steel kitchen",
    ),
    (
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "Two chefs in a restaurant kitchen preparing food.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "Two cooks are cooking the food someone ordered at this restaurant",
    ),
    (
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "The chef is cooking with pans on the stove next to an oven.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "Two men that are standing in a kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000222564.jpg",
        "Two cooks are near the stove in a stainless steel kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "this is a very dark picture of a room with a shelf",
    ),
    (
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "a cluttered room with a table and shelf on the wall.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "A view of a messy room, with shelves on the wall.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "A dark and cluttered storage area with wood walls.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000118113.jpg",
        "A dim lit room consisting of many objects put together.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "A kitchen filled with black appliances and lots of counter top space.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "some brown cabinets a black oven a tea kettle and a microwave",
    ),
    (
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "A small kitchen with glass and wooden cabinets.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "A modern style kitchen filled with may different items.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000193271.jpg",
        "A kitchen with wooden cabinets and black appliances.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "A professional kitchen filled with sinks and appliances.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "A kitchen area with toilet and various cleaning appliances.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "A commercial dish washing station with a toilet in it.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "A toilet and mop bucket in a kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000224736.jpg",
        "A cluttered room with a sink, a toilet and in industrial mop bucket.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "A man on a bicycle riding next to a train",
    ),
    (
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "A person is riding a bicycle but there is a train in the background.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "a red and white train and a man riding a bicycle",
    ),
    (
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "a guy that is riding his bike next to a train",
    ),
    (
        "http://images.cocodataset.org/train2017/000000483108.jpg",
        "A man riding a bike past a train traveling along tracks.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "A narrow kitchen filled with appliances and cooking utensils.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "A galley kitchen with cabinets and appliances on both sides",
    ),
    (
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "A hallway leading into a white kitchen with appliances.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "Doorway view of a kitchen with a sink, stove, refrigerator and pantry.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000403013.jpg",
        "The pantry door of the small kitchen is closed.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "A kitchen with wood floors and lots of furniture.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "A beautiful, open kitchen and dining room area features an island in the center and wood cabinets and large windows.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "A kitchen made of mostly wood with a small desk with a laptop.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "A very spacious room with a kitchen and dining area.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000374628.jpg",
        "A full view of an open kitchen and dining area.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "A woman eating vegetables in front of a stove.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "A woman forks vegetables out of a bowl into her mouth.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "Woman eating an assortment of mixed vegetables in a bowl.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "A young woman standing in a kitchen eats a plate of vegetables.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000328757.jpg",
        "A woman eating fresh vegetables from a bowl.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "A kitchen is shown with a variety of items on the counters.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "A kitchen has the windows open and plaid curtains.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "A kitchen with two windows and two metal sinks.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "An older kitchen with cluttered counter tops but empty sink.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000384213.jpg",
        "Glasses and bottles are placed near a kitchen sink.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "A boy performing a kickflip on his skateboard on a city street.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "A man is doing a trick on a skateboard",
    ),
    (
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "A guy jumps in the air with his skateboard beneath him.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "Man in all black doing a trick on his skateboard.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000293802.jpg",
        "A skateboarder flipping his board on a street.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "A kitchen with a stove, microwave and refrigerator.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "A refrigerator, oven and microwave sitting in a kitchen.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "The kitchenette uses small space to great efficiency.",
    ),
    (
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "an image of a kitchen setting with black appliances",
    ),
    (
        "http://images.cocodataset.org/train2017/000000086408.jpg",
        "A kitchen with cabinets, a stove, microwave and refrigerator.",
    ),
]


def get_data(multi_caption=False):
    if os.path.exists(os.path.join("data", "data_dump.pkl")):
        data = pkl.load(open(os.path.join("data", "data_dump.pkl"), "rb"))
        return data

    data = []
    urls = set()

    for url, caption in tqdm(_data_pair, desc="Downloading images"):
        if not multi_caption and url in urls:
            continue
        urls.add(url)
        image = Image.open(requests.get(url, stream=True).raw)
        if image.mode != "RGB":
            continue
        data.append({"image": image, "target": caption, "uuid": str(uuid.uuid1())})
    pkl.dump(data, open(os.path.join("data", "data_dump.pkl"), "wb"))
    return data
