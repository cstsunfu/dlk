# # from transformers import BertTokenizer
# import json

# from tokenizers import Tokenizer

# token = json.dumps(json.load(open("./data/bert/tokenizer.json", "r")))
# tokenizer = Tokenizer.from_str(token)

# a = ["nice to meet you a b c d e f g h i j k"]

# # return input_ids, attention_mask, token_type_ids, word_ids, special_tokens_mask
# tokens = tokenizer.encode_batch(
#     a,
#     # padding="longest",
#     # truncation=True,
#     # max_length=64,
# )

# print(tokens[0].sequence_ids)


word_offset_a_list = [1, 2]
word_offset_b_list = [1, 2]
offsets_list = [1, 2]
word_ids_list = [1, 2]
type_ids_list = [1, 2]

fixed_offsets_list = []
for word_offset_a, word_offset_b, offsets, word_ids, type_ids in zip(
    word_offset_a_list,
    word_offset_b_list,
    offsets_list,
    word_ids_list,
    type_ids_list,
):
    print(word_offset_a)
    # fixed_offsets = []
    # word_offsets = [word_offset_a, word_offset_b]
    # for offset, word_id, type_id in zip(offsets, word_ids, type_ids):
    #     if offset == (0, 0):
    #         fixed_offsets.append(offset)
    #     else:
    #         fixed_offsets.append(
    #             (
    #                 offset[0] + word_offsets[type_id][word_id][0],
    #                 offset[1] + word_offsets[type_id][word_id][0],
    #             )
    #         )
    # fixed_offsets_list.append(fixed_offsets)
# return fixed_offsets_list
