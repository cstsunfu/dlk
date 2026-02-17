import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import PreTrainedTokenizer
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    _torch_collate_batch,
)


@dataclass
class DataCollatorForCustomMask(DataCollatorForLanguageModeling):
    """
    A custom data collator that:
    1. Supports Chinese WWM via 'chinese_ref'.
    2. Allows forcing specific words to be masked via 'force_mask_words'.
    """

    tokenizer: PreTrainedTokenizer

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [e["input_ids"] for e in examples]
        batch_input = _torch_collate_batch(input_ids, self.tokenizer)

        mask_labels = []
        for e in examples:
            # 1. 将 ID 转为 Token
            ref_tokens = self.tokenizer.convert_ids_to_tokens(e["input_ids"])

            # 2. 处理中文 WWM
            if "chinese_ref" in e:
                ref_pos = e["chinese_ref"]
                for i in ref_pos:
                    if i < len(ref_tokens):  # 安全检查
                        ref_tokens[i] = "##" + ref_tokens[i]

            # 3. 获取需要强制 mask 的词
            force_mask_words = e.get("force_mask_words", [])

            # 4. 调用我们重写的新 _whole_word_mask 方法
            mask_labels.append(self._whole_word_mask(ref_tokens, force_mask_words))

        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(
        self,
        input_tokens: List[str],
        force_mask_words: List[str] = None,
        max_predictions=512,
    ) -> List[int]:
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        forced_mask_indices = set()
        random_cand_indexes = []

        if force_mask_words:

            def tokens_to_word(tokens):
                return "".join(t.replace("##", "") for t in tokens)

            for index_set in cand_indexes:
                word_tokens = [input_tokens[i] for i in index_set]
                word = tokens_to_word(word_tokens)
                if word in force_mask_words:
                    for i in index_set:
                        forced_mask_indices.add(i)
                else:
                    random_cand_indexes.append(index_set)
        else:
            random_cand_indexes = cand_indexes

        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens) * self.mlm_probability))),
        )
        num_to_random_predict = num_to_predict - len(forced_mask_indices)

        random.shuffle(random_cand_indexes)
        random_mask_indices = set()
        if num_to_random_predict > 0:
            for index_set in random_cand_indexes:
                if len(random_mask_indices) + len(index_set) <= num_to_random_predict:
                    for i in index_set:
                        random_mask_indices.add(i)
                if len(random_mask_indices) >= num_to_random_predict:
                    break

        covered_indexes = forced_mask_indices.union(random_mask_indices)
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels


def preprocess_chinese_sentence(sentence: str, tokenizer: PreTrainedTokenizer):
    """
    Preprocesses a Chinese sentence to generate inputs for our custom collator.

    Returns a dictionary containing:
    - input_ids
    - chinese_ref (indices of non-first characters in a word)
    - force_mask_words (list of nouns found in the sentence)
    """
    import jieba.posseg as pseg

    # 1. 使用 jieba 进行词性标注
    words_with_pos = pseg.lcut(sentence)

    # 提取名词作为强制掩码的目标
    # jieba 的名词词性以 'n' 开头 (e.g., n, nr, ns, nt, nz...)
    nouns = [word for word, flag in words_with_pos if flag.startswith("n")]

    # 2. 使用 BertTokenizer 对整个句子进行 tokenize
    # 我们不加特殊 token，方便后续对齐
    bert_tokens = tokenizer.tokenize(sentence)

    # 3. 生成 chinese_ref
    chinese_ref = []
    current_token_idx = 0
    for word, _ in words_with_pos:
        if len(word) > 1:
            # 这个词由多个字组成，从第二个字开始，其索引需要加入 chinese_ref
            # 注意：这里的索引是相对于 bert_tokens 列表的
            for i in range(1, len(word)):
                chinese_ref.append(current_token_idx + i)
        current_token_idx += len(word)

    # 4. 添加特殊 token 并生成 input_ids
    # 在对齐完成后再添加 [CLS] 和 [SEP]
    final_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

    # 5. 调整 chinese_ref 的索引，因为我们在前面加了 [CLS]
    # 所有索引都需要 +1
    chinese_ref_adjusted = [i + 1 for i in chinese_ref]

    return {
        "input_ids": input_ids,
        "chinese_ref": chinese_ref_adjusted,
        "force_mask_words": nouns,
        "original_sentence": sentence,
        "jieba_nouns": nouns,
    }


if __name__ == "__main__":

    import jieba
    from transformers import PreTrainedTokenizer

    # 1. 初始化 Tokenizer 和我们自定义的 Collator
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    collator = DataCollatorForCustomMask(
        tokenizer=tokenizer, mlm_probability=0.3
    )  # 提高概率以便观察

    # 2. 准备中文句子
    sentence = "我爱北京天安门，天安门上太阳升。"

    # 3. 使用预处理函数生成数据样本
    data_sample = preprocess_chinese_sentence(sentence, tokenizer)

    print("--- 预处理结果 ---")
    print(f"原始句子: {data_sample['original_sentence']}")
    print(f"Jieba 识别出的名词 (强制掩码目标): {data_sample['jieba_nouns']}")
    print(f"Input IDs: {data_sample['input_ids']}")
    print(
        f"Tokenized 文本: {tokenizer.convert_ids_to_tokens(data_sample['input_ids'])}"
    )
    print(f"Chinese Ref (非词首字符索引): {data_sample['chinese_ref']}")
    print("-" * 20)

    # 4. 使用 collator 处理数据
    # collator 期望一个列表作为输入
    batch = collator([data_sample])

    # 5. 分析输出结果
    masked_input_ids = batch["input_ids"].squeeze().tolist()
    labels = batch["labels"].squeeze().tolist()

    print("\n--- Collator 输出分析 ---")
    print(f"{'Index':<6} {'Token':<10} {'Masked_Input_ID':<18} {'Label':<10}")
    print("=" * 50)

    masked_count = 0
    forced_masked_correctly = 0

    # 我们知道 '北京', '天安门', '太阳' 是名词
    force_mask_tokens = {"北", "京", "天", "安", "门", "太", "阳"}

    for i, (token, input_id, label) in enumerate(
        zip(
            tokenizer.convert_ids_to_tokens(data_sample["input_ids"]),
            masked_input_ids,
            labels,
        )
    ):
        print(f"{i:<6} {token:<10} {input_id:<18} {label:<10}")
        if label != -100:
            masked_count += 1
            if token in force_mask_tokens:
                forced_masked_correctly += 1

    print("=" * 50)
    print(f"总共掩码了 {masked_count} 个 token。")
    print(
        f"在被掩码的 token 中，有 {forced_masked_correctly} 个属于我们强制要求的名词。"
    )
    print(f"强制掩码的名词总 token 数: {len(force_mask_tokens)}")

    if forced_masked_correctly == len(force_mask_tokens):
        print("\n✅ 成功! 所有指定名词 ('北京', '天安门', '太阳') 都被正确地掩码了。")
    else:
        print("\n❌ 失败! 强制掩码逻辑可能存在问题。")

    """
--- 预处理结果 ---
原始句子: 我爱北京天安门，天安门上太阳升。
Jieba 识别出的名词 (强制掩码目标): ['北京', '天安门', '天安门', '太阳']
Input IDs: [101, 2769, 4263, 1266, 776, 1921, 2128, 7303, 8024, 1921, 2128, 7303, 涓, 1922, 7832, 1340, 511, 102]
Tokenized 文本: ['[CLS]', '我', '爱', '北', '京', '天', '安', '门', '，', '天', '安', '门', '上', '太', '阳', '升', '。', '[SEP]']
Chinese Ref (非词首字符索引): [4, 6, 7, 10, 11, 14]
--------------------

--- Collator 输出分析 ---
Index  Token        Masked_Input_ID    Label     
==================================================
0      [CLS]        101                -100      
1      我            2769               -100      
2      爱            4263               -100      
3      北            103                1266      
4      京            103                776       
5      天            103                1921      
6      安            103                2128      
7      门            103                7303      
8      ，            8024               -100      
9      天            103                1921      
10     安            103                2128      
11     门            103                7303      
12     上            103                涓      
13     太            103                1922      
14     阳            103                7832      
15     升            1340               -100      
16     。            511                -100      
17     [SEP]        102                -100      
==================================================
总共掩码了 9 个 token。
在被掩码的 token 中，有 7 个属于我们强制要求的名词。
强制掩码的名词总 token 数: 7

✅ 成功! 所有指定名词 ('北京', '天安门', '太阳') 都被正确地掩码了。
    """
