import hjson
import pickle as pkl
import os
import json
from typing import Union, Dict, Any
from dlk.utils.parser import BaseConfigParser, PostProcessorConfigParser
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.logger import Logger
import torch
from dlk.utils.vocab import Vocabulary
import torchmetrics

logger = Logger.get_logger()


@postprocessor_config_register('txt_cls')
class TxtClsPostProcessorConfig(IPostProcessorConfig):
    """docstring for TxtClsPostProcessorConfig
    config e.g.
    {
        "_name": "txt_cls",
        "config": {
            "meta": "*@*",
            "meta_data": {
                "label_vocab": 'label_vocab',
            },
            "input_map": {
                "logits": "logits",
                "label_ids": "label_ids"
                "_index": "_index",
            },
            "origin_input_map": {
                "sentence": "sentence",
                "sentence_a": "sentence_a", // for pair
                "sentence_b": "sentence_b",
                "uuid": "uuid"
            },
            "save_root_path": ".",  //save data root dir
            "top_k": 1, //the result return top k result
            "focus": [], //always return the list label values which index in 'focus' list, if the focus[0] == 'pos', then the predict value always return the logits[vocab.get_index('pos')], and the label always be 'pos'
            "data_type": "single", //single or pair
            "save_path": {
                "valid": "valid",  // relative dir for valid stage
                "test": "test",    // relative dir for test stage
            },
            "start_save_step": 0,  // -1 means the last
            "start_save_epoch": -1,
        }
    }
    """

    def __init__(self, config: Dict):
        super(TxtClsPostProcessorConfig, self).__init__(config)

        self.data_type = self.config['data_type']
        assert self.data_type in {'single', 'pair'}
        if self.data_type == 'pair':
            self.sentence_a = self.origin_input_map['sentence_a']
            self.sentence_b = self.origin_input_map['sentence_b']
        else:
            self.sentence = self.origin_input_map['sentence']
        self.uuid = self.origin_input_map['uuid']

        self.logits = self.input_map['logits']
        self.label_ids = self.input_map['label_ids']
        self._index = self.input_map['_index']
        self.top_k = self.config['top_k']
        self.focus = self.config['focus']
        if isinstance(self.config['meta'], str):
            meta = pkl.load(open(self.config['meta'], 'rb'))
        else:
            raise PermissionError("You must provide meta data for txt_cls postprocess.")
        trace_path = []
        trace_path_str = self.config['meta_data']['label_vocab']
        if trace_path_str and trace_path_str.strip()!='.':
            trace_path = trace_path_str.split('.')
        self.label_vocab = meta
        for trace in trace_path:
            self.label_vocab = self.label_vocab[trace]
        self.label_vocab = Vocabulary.load(self.label_vocab)
        self.save_path = self.config['save_path']
        self.save_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']
        self.post_check(self.config, used=[
            "meta",
            "meta_data",
            "input_map",
            "origin_input_map",
            "save_root_path",
            "top_k",
            "focus",
            "data_type",
            "save_path",
            "start_save_step",
            "start_save_epoch",
        ])


@postprocessor_register('txt_cls')
class TxtClsPostProcessor(IPostProcessor):
    """docstring for TxtClsPostProcessor"""
    def __init__(self, config: TxtClsPostProcessorConfig):
        super(TxtClsPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.acc_calc = torchmetrics.Accuracy()

    def do_predict(self, stage, list_batch_outputs, origin_data, rt_config):
        """TODO: Docstring for do_predict.
        :stage: TODO
        :list_batch_outputs: the batch_output means the input
        :origin_data: the origin_data means the origin_input
        :rt_config: TODO
        :returns: TODO
        """
        results = []
        for outputs in list_batch_outputs:
            logits = outputs[self.config.logits].detach()
            assert len(logits.shape) == 2
            # predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs[self.config._index])

            if self.config.label_ids in outputs:
                label_ids = outputs[self.config.label_ids]
            else:
                label_ids = [None] * len(indexes)
            for one_logits, index, label_id in zip(logits, indexes, label_ids):
                one_ins = {}
                one_logits = torch.softmax(one_logits, -1)
                one_origin = origin_data.iloc[int(index)]
                if self.config.data_type == 'single':
                    sentence = one_origin[self.config.sentence]
                    one_ins['sentence'] = sentence
                else:
                    sentence_a = one_origin[self.config.sentence_a]
                    one_ins['sentence_a'] = sentence_a
                    sentence_b = one_origin[self.config.sentence_b]
                    one_ins['sentence_b'] = sentence_b
                uuid = one_origin[self.config.uuid]
                if self.config.focus:
                    label_values, label_indeies = [], []
                    for label_name in self.config.focus:
                        label_index = self.label_vocab.get_index(label_name)
                        label_values.append(one_logits[label_index])
                        label_indeies.append(label_index)
                else:
                    label_values, label_indeies = torch.topk(one_logits, self.config.top_k, dim=-1)
                predict = []
                for label_value, label_index in zip(label_values, label_indeies):
                    label_name = self.label_vocab.get_word(label_index)
                    predict.append([label_name, float(label_value)])
                if label_id is None:
                    ground_truth = ""
                else:
                    ground_truth = self.label_vocab.get_word(label_id)
                one_ins['uuid'] = uuid
                one_ins['labels'] = ground_truth
                one_ins['predict_labels'] = predict
                results.append(one_ins)
        return results

    def do_calc_metrics(self, predicts, stage, list_batch_outputs, origin_data, rt_config):
        """TODO: Docstring for do_calc_metrics.
        :returns: TODO

        """
        for outputs in list_batch_outputs:
            logits = outputs[self.config.logits]
            label_ids = outputs[self.config.label_ids].squeeze(-1)
            self.acc_calc.update(logits, label_ids)
        real_name = self.loss_name_map(stage)
        return {f'{real_name}_acc': self.acc_calc.compute()}

    def do_save(self, predicts, stage, list_batch_outputs, origin_data, rt_config={}, save_condition=False):
        """TODO: Docstring for do_save.

        :predicts: TODO
        :rt_config: TODO
        :condition: when the save condition is True, do save
        :returns: TODO
        """
        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config.get('total_steps', 0) - 1
            self.config.start_save_epoch = rt_config.get('total_epochs', 0) - 1
        if rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch:
            save_condition = True
        if save_condition:
            save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if "current_step" in rt_config:
                save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}_predict.json")
            else:
                save_file = os.path.join(save_path, 'predict.json')
            logger.info(f"Save the {stage} predict data at {save_file}")
            json.dump(predicts, open(save_file, 'w'), indent=4)
