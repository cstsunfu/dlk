'''
'''
import hjson
import pickle as pkl
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from typing import Union, Dict
from dlkit.utils.parser import BaseConfigParser, PostProcessorConfigParser
from dlkit.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlkit.utils.config import ConfigTool
from dlkit.utils.logger import logger
import torch
from dlkit.utils.vocab import Vocabulary
from tokenizers import Tokenizer
import torchmetrics


logger = logger()

        

@postprocessor_config_register('ner')
class NerPostProcessorConfig(IPostProcessorConfig):
    """docstring for IdentityPostProcessorConfig
    config e.g.
    {
        "_name": "ner",
        "config": {
            "meta": "*@*",
            "meta_data": {
                "label_vocab": 'label_vocab',
                "tokenizer": "tokenizer",
            },
            "output_data": {
                "logits": "logits",
                "label_id": "label_id",
                "input_ids": "input_ids",
                "special_tokens_mask": "special_tokens_mask",
                "attention_mask": "attention_mask",
            },
            "origin_data": {
                "sentence": "sentence",
                "offsets": "offsets"
            },
            "save_root_path": ".",  //save data root dir
            "save_path": {
                "valid": "valid",  // relative dir for valid stage
                "test": "test",    // relative dir for test stage
            },
            "start_save_step": 0,  // -1 means the last
            "start_save_epoch": -1,
            "aggregation_strategy": "max", // AggregationStrategy item
            "ignore_labels": ['O', 'X', 'S', "E"], // Out, Out, Start, End
        }
    }
    """

    def __init__(self, config: Dict):
        super(NerPostProcessorConfig, self).__init__(config)

        self.logits = self.output_data['logits']
        self.sentence = self.config['origin_data']['sentence']
        self.offsets = self.config['origin_data']['offsets']
        self.aggregation_strategy = self.config['aggregation_strategy']
        self.label_id = self.output_data['label_id']
        self.ignore_labels = self.output_data['ignore_labels']
        self.input_ids = self.output_data['input_ids']
        self.special_tokens_mask = self.output_data['special_tokens_mask']
        self.attention_mask = self.output_data['attention_mask']
        if isinstance(self.config['meta'], str):
            meta = pkl.load(open(self.config['meta'], 'rb'))
        else:
            raise PermissionError("You must provide meta data(vocab & tokenizer) for ner postprocess.")

        vocab_trace_path = []
        vocab_trace_path_str = self.config['meta_data']['label_vocab']
        if vocab_trace_path_str and vocab_trace_path_str.strip()!='.':
            vocab_trace_path = vocab_trace_path_str.split('.')
        assert vocab_trace_path, "We need vocab and tokenizer all in meta, so you must provide the trace path from meta"
        self.label_vocab = meta
        for trace in vocab_trace_path:
            self.label_vocab = self.label_vocab[trace]
        self.label_vocab = Vocabulary.load(self.label_vocab)


        tokenizer_trace_path = []
        tokenizer_trace_path_str = self.config['meta_data']['tokenizer']
        if tokenizer_trace_path_str and tokenizer_trace_path_str.strip()!='.':
            tokenizer_trace_path = tokenizer_trace_path_str.split('.')
        assert tokenizer_trace_path, "We need vocab and tokenizer all in meta, so you must provide the trace path from meta"
        tokenizer_config = meta
        for trace in tokenizer_trace_path:
            tokenizer_config = tokenizer_config[trace]
        self.tokenizer = Tokenizer.from_str(tokenizer_config)

        self.save_path = self.config['save_path']
        self.save_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']


class AggregationStrategy(object):
    """docstring for AggregationStrategy"""
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"
        

@postprocessor_register('ner')
class NerPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: NerPostProcessorConfig):
        super(NerPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.tokenizer = self.config.tokenizer
        self.metric = torchmetrics.Accuracy()


    def process(self, stage, outputs, origin_data, rt_config)->Dict:
        """ This script is mostly copied from Transformers
        rt_config={
            "current_step": self.global_step,
            "current_epoch": self.current_epoch, 
            "total_steps": self.num_training_steps, 
            "total_epochs": self.num_training_epochs
        }),
        :returns: TODO

        """
        batch_logits = outputs[self.config.logits]
        batch_special_tokens_mask = outputs[self.config.special_tokens_mask]
        batch_attention_mask = outputs[self.config.attention_mask]

        if self.config.label_id in outputs:
            batch_label_ids = list(outputs[self.config.label_id])
        else:
            batch_label_ids = batch_logits.shape[0] * [[]]

        indexes = list(outputs["_index"])

        batch_input_ids = outputs[self.config.input_ids]
        outputs = []

        for logits, index, label_ids, input_ids, special_tokens_mask, attention_mask in zip(batch_logits, indexes, batch_label_ids, batch_input_ids, batch_special_tokens_mask, batch_attention_mask):
            one_ins = {}
            sentence = origin_data.iloc[int(index)][self.config.sentence]
            offset_mapping = origin_data.iloc[int(index)][self.config.offsets]
            logits = logits.numpy()
            special_tokens_mask = special_tokens_mask.numpy()

            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
            pre_entities = self.gather_pre_entities(
                sentence, input_ids, scores, offset_mapping, special_tokens_mask, self.config.aggregation_strategy
            )
            grouped_entities = self.aggregate(pre_entities, self.config.aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in self.config.ignore_labels
                and entity.get("entity_group", None) not in self.config.ignore_labels
            ]
            one_ins['sentence'] = sentence
            one_ins['entities'] = entities
            outputs.append(one_ins)

        save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}.csv")
        logger.info(f"Save the {stage} predict data at {save_file}")
        json.dump(outputs, open(save_file, 'w'), indent=4)
        # TODO Metrics
        log_info = {}
        log_info["acc"] = self.metric(logits, outputs[self.config.label_id])
        return log_info


    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.config.label_vocab[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.label_vocab[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.label_vocab[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.config.label_vocab[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity


    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups

    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer.model, "continuing_subword_prefix", None):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    is_subword = sentence[start_ind - 1 : start_ind] != " " if start_ind > 0 else False

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

