import hjson
import pickle as pkl
import os
import json
from typing import Union, Dict, Any
from dlkit.utils.parser import BaseConfigParser, PostProcessorConfigParser
from dlkit.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlkit.utils.config import ConfigTool
from dlkit.utils.logger import logger
import torch
from dlkit.utils.vocab import Vocabulary
import torchmetrics

logger = logger()


@postprocessor_config_register('txt_cls')
class TxtClsPostProcessorConfig(IPostProcessorConfig):
    """docstring for IdentityPostProcessorConfig
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
                "uuid": "uuid"
            },
            "save_root_path": ".",  //save data root dir
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

        self.sentence = self.origin_input_map['sentence']
        self.uuid = self.origin_input_map['uuid']

        self.logits = self.input_map['logits']
        self.label_ids = self.input_map['label_ids']
        self._index = self.input_map['_index']
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


@postprocessor_register('txt_cls')
class TxtClsPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: TxtClsPostProcessorConfig):
        super(TxtClsPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.acc_calc = torchmetrics.Accuracy()

    def calc_metrics(self, list_batch_outputs, stage):
        """TODO: Docstring for calc_metrics.
        :returns: TODO

        """
        for outputs in list_batch_outputs:
            logits = outputs[self.config.logits]
            label_ids = outputs[self.config.label_ids]
            self.acc_calc.update(logits, label_ids)
        real_name = self.loss_name_map(stage)
        return {f'{real_name}_acc': self.acc_calc.compute()}
        

    def process(self, stage, list_batch_outputs, origin_data, rt_config)->Dict:
    # def process(self, stage, outputs, origin_input_map, rt_config: Dict[str, Any])->Dict:
        """TODO: Docstring for process.
        :data: TODO
         rt_config={
             "current_step": self.global_step,
             "current_epoch": self.current_epoch, 
             "total_steps": self.num_training_steps, 
             "total_epochs": self.num_training_epochs
         }),
        :returns: TODO

        """
        # logits = outputs[self.config.logits]

        log_info = {}
        log_info[self.loss_name_map(stage)] = self.average_loss(list_batch_outputs=list_batch_outputs)

        metrics = {}
        if stage not in self.without_ground_truth_stage:
            metrics = self.calc_metrics(list_batch_outputs, stage)
        log_info.update(metrics)

        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config['total_steps'] - 1
            self.config.start_save_epoch = rt_config['total_epochs'] - 1
        if rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch:

            result = self.predict(list_batch_outputs, origin_data)

            save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}_predict.json")
            logger.info(f"Save the {stage} predict data at {save_file}")
            json.dump(result, open(save_file, 'w'), indent=4)
        return log_info

    def predict(self, list_batch_outputs, origin_data):
        """TODO: Docstring for predict.

        :list_batch_outputs: TODO
        :returns: TODO

        """
        result = []
        for outputs in list_batch_outputs:
            logits = outputs[self.config.logits]
            assert len(logits.shape) == 2
            predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs[self.config._index])

            if self.config.label_ids in outputs:
                label_ids = outputs[self.config.label_ids]
            else:
                label_ids = [None] * len(indexes)
            for predict_index, index, label_id in zip(predict_indexes, indexes, label_ids):
                one_ins = {}
                origin_data = origin_data.iloc[int(index)]
                sentence = origin_data[self.config.sentence]
                uuid = origin_data[self.config.uuid]
                predict = self.label_vocab.get_word(predict_index)
                if label_id is None:
                    ground_truth = ""
                else:
                    ground_truth = self.label_vocab.get_word(label_id)
                one_ins['uuid'] = uuid
                one_ins['sentence'] = sentence
                one_ins['ground_truth'] = ground_truth
                one_ins['predict'] = predict
                result.append(one_ins)
        return result
