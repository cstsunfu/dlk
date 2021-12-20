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


@postprocessor_config_register('txt_reg')
class TxtRegPostProcessorConfig(IPostProcessorConfig):
    """docstring for TxtRegPostProcessorConfig
    config e.g.
    {
        "_name": "txt_reg",
        "config": {
            "input_map": {
                "logits": "logits",
                "values": "values",
                "_index": "_index",
            },
            "origin_input_map": {
                "sentence": "sentence",
                "sentence_a": "sentence_a", // for pair
                "sentence_b": "sentence_b",
                "uuid": "uuid"
            },
            "data_type": "single", //single or pair
            "save_root_path": ".",  //save data root dir
            "save_path": {
                "valid": "valid",  // relative dir for valid stage
                "test": "test",    // relative dir for test stage
            },
            "log_reg": false, // whether logistic regression
            "start_save_step": 0,  // -1 means the last
            "start_save_epoch": -1,
        }
    }
    """

    def __init__(self, config: Dict):
        super(TxtRegPostProcessorConfig, self).__init__(config)

        self.data_type = self.config['data_type']
        assert self.data_type in {'single', 'pair'}
        if self.data_type == 'pair':
            self.sentence_a = self.origin_input_map['sentence_a']
            self.sentence_b = self.origin_input_map['sentence_b']
        else:
            self.sentence = self.origin_input_map['sentence']
        self.uuid = self.origin_input_map['uuid']
        self.log_reg = self.config['log_reg']

        self.value = self.input_map['values']
        self.logits = self.input_map['logits']
        self._index = self.input_map['_index']
        self.save_path = self.config['save_path']
        self.save_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']
        self.post_check(self.config, used=[
            "input_map",
            "origin_input_map",
            "save_root_path",
            "save_path",
            "data_type",
            "start_save_step",
            "start_save_epoch",
            "log_reg",
        ])


@postprocessor_register('txt_reg')
class TxtRegPostProcessor(IPostProcessor):
    """docstring for TxtRegPostProcessor"""
    def __init__(self, config: TxtRegPostProcessorConfig):
        super(TxtRegPostProcessor, self).__init__()
        self.config = config

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
            if self.config.log_reg:
                logits = torch.sigmoid(logits)
            assert len(logits.shape) == 2
            # predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs[self.config._index])

            if self.config.value in outputs:
                values = outputs[self.config.value]
            else:
                values = [0.0] * len(indexes)
            for one_logits, index, value in zip(logits, indexes, values):
                one_ins = {}
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
                one_ins['uuid'] = uuid
                one_ins['values'] = [float(value)]
                one_ins['predict_values'] = [float(one_logits)]
                results.append(one_ins)
        return results

    def do_calc_metrics(self, predicts, stage, list_batch_outputs, origin_data, rt_config):
        """TODO: Docstring for do_calc_metrics.
        :returns: TODO

        """
        return {}

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
