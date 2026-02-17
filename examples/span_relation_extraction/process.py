# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""
data format
{
    "uuid": "",
    "sentence": "",
    "entities_info": [
        {
            "entity_id": "bd798da3-52a8-11ed-801c-18c04d299e80",
            "start":1,
            "end": 3,
            "labels": [
                "Brand"
            ]
        },
        {
            "entity_id": "bd7a2928-52a8-11ed-8b5c-18c04d299e80",
            "start":5,
            "end": 9,
            "labels": [
                "Product"
            ]
        }
    ],
    "relations_info": [
        {
            "labels": [
                "belong_to"
            ],
            "from": "bd7a2928-52a8-11ed-8b5c-18c04d299e80", # id of entity
            "to": "bd798da3-52a8-11ed-801c-18c04d299e80",
        }
    ]
}
"""
import json
import logging
import uuid
from typing import Any, Dict, List

from datasets import concatenate_datasets, load_dataset

from dlk.preprocess import PreProcessor

logger = logging.getLogger(__name__)


def process_batch(batch):
    """
    CoNLL04 数据处理核心逻辑：
    1. DFKI-SLT/conll04 数据集包含 ['entities', 'tokens', 'relations', 'orig_id']。
    2. 将 tokens 拼接为 sentence，并严格计算每个 token 对应的 character offset（字符级别起始与结束索引）。
    3. 将基于 token index 的 entities 的 start 和 end，转换为基于 character index 的 start 和 end。
    4. 生成实体的唯一 id (entity_id)，并在 relations_info 中用 entity_id 来代替基于索引的 head 和 tail。
    """
    batch_uuids = []
    batch_sentences = []
    batch_entities_info = []
    batch_relations_info = []

    # 遍历 batch 中的每一条数据
    for idx, tokens in enumerate(batch["tokens"]):
        entities = batch["entities"][idx]
        relations = batch["relations"][idx]

        # 1. 拼接句子并记录 token -> character 级别的 offset
        token_char_offsets = []
        current_offset = 0
        for token in tokens:
            token_length = len(token)
            # 记录当前 token 的字符级 (start, end)
            token_char_offsets.append((current_offset, current_offset + token_length))
            # 加上一个空格的长度
            current_offset += token_length + 1

        sentence = " ".join(tokens)

        # 2. 处理实体信息
        entities_info = []
        # 保存 Dataset 原本索引 -> 我们自己生成的 UUID 映射，用于之后处理 relations
        entity_idx_to_id = {}

        for e_idx, ent in enumerate(entities):
            # CoNLL04 数据集中的 entities 格式如: {"start": 0, "end": 2, "type": "Org"}
            # 其 end 是开区间 (exclusive)的。即 start=0, end=2 包含第0和第1个 token。
            t_start = ent["start"]
            t_end = ent["end"] - 1  # 转为闭区间来取索引

            # 安全检查防越界
            t_end = min(t_end, len(token_char_offsets) - 1)

            # 拿到字符级别的 start 和 end
            char_start = token_char_offsets[t_start][0]
            char_end = token_char_offsets[t_end][1]

            ent_id = str(uuid.uuid4())
            entity_idx_to_id[e_idx] = ent_id

            entities_info.append(
                {
                    "entity_id": ent_id,
                    "start": char_start,  # 框架要求：字符级别 start
                    "end": char_end,  # 框架要求：字符级别 end
                    "labels": [ent["type"]],  # 实体标签
                }
            )

        # 3. 处理关系信息
        relations_info = []
        for rel in relations:
            # CoNLL04 的 relations 格式如: {"head": 0, "tail": 1, "type": "Work_For"}
            head_idx = rel["head"]
            tail_idx = rel["tail"]
            rel_type = rel["type"]

            # 确保引用的实体存在
            if head_idx in entity_idx_to_id and tail_idx in entity_idx_to_id:
                relations_info.append(
                    {
                        "from": entity_idx_to_id[head_idx],
                        "to": entity_idx_to_id[tail_idx],
                        "labels": [rel_type],
                    }
                )

        # 原数据集如果有 orig_id 则保留，没有则自动生成
        ins_uuid = (
            str(batch["orig_id"][idx]) if "orig_id" in batch else str(uuid.uuid4())
        )

        batch_uuids.append(ins_uuid)
        batch_sentences.append(sentence)
        batch_entities_info.append(entities_info)
        batch_relations_info.append(relations_info)

    # uuid, sentence, entities_info, relations_info
    return {
        "uuid": batch_uuids,
        "sentence": batch_sentences,
        "entities_info": batch_entities_info,
        "relations_info": batch_relations_info,
    }


if __name__ == "__main__":
    # 1. 加载权威数据集
    logger.info("Loading DFKI-SLT/conll04 dataset...")
    try:
        dataset = load_dataset("DFKI-SLT/conll04")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        exit(1)

    # 2. 获取原始列名，准备清除以避免后续冲突
    raw_cols = dataset[list(dataset.keys())[0]].column_names

    # 3. 数据处理 (使用 dataset.map 加速并规范化数据结构)
    logger.info("Mapping raw dataset to Span-Relation expected format...")
    processed_datasets = dataset.map(
        process_batch, batched=True, remove_columns=raw_cols, desc="Processing dataset"
    )

    # 4. 构建 Input Data 字典
    # "train" set merge train and validataion processed_datasets["train"] + processed_datasets["validation"]
    input_data = {
        "train": concatenate_datasets(
            [processed_datasets["train"], processed_datasets["validation"]]
        ),
        "valid": processed_datasets["test"],
    }

    # "test": processed_datasets["test"]
    # 5. 运行 PreProcessor
    logger.info("Running PreProcessor to generate tokenized data and vocabs...")
    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)

    logger.info("Data processing complete!")
