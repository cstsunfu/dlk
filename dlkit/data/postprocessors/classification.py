import hjson
import pickle as pkl
import pandas as pd
import os
from typing import Union, Dict
from dlkit.utils.parser import BaseConfigParser, PostProcessorConfigParser
from dlkit.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlkit.utils.config import ConfigTool
from dlkit.utils.logger import logger
import torch
from dlkit.utils.vocab import Vocabulary
import torchmetrics

# initialize metric

logger = logger()

        

@postprocessor_config_register('classification')
class ClassificationPostProcessorConfig(IPostProcessorConfig):
    """docstring for IdentityPostProcessorConfig
    config e.g.
    {
        "_name": "classification",
        "config": {
            "meta": "*@*",
            "meta_data": {
                "label_vocab": 'label_vocab',
            },
            "output_data": {
                "logits": "logits",
                "label_ids": "label_ids"
            },
            "origin_data": {
                "sentence": "sentence"
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
        super(ClassificationPostProcessorConfig, self).__init__(config)

        self.logits = self.output_data['logits']
        self.sentence = self.config['origin_data']['sentence']
        self.label_ids = self.output_data['label_ids']
        if isinstance(self.config['meta'], str):
            meta = pkl.load(open(self.config['meta'], 'rb'))
        else:
            raise PermissionError("You must provide meta data for classification postprocess.")
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


@postprocessor_register('classification')
class ClassificationPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: ClassificationPostProcessorConfig):
        super(ClassificationPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.metric = torchmetrics.Accuracy()

    def process(self, stage, outputs, origin_data, rt_config)->Dict:
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
        logits = outputs[self.config.logits]

        log_info = {}
        if self.config.label_ids in outputs:
            log_info["acc"] = self.metric(logits, outputs[self.config.label_ids])
            label_ids = outputs[self.config.label_ids]
        else:
            label_ids = [-1]*logits.shape[0]

        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config['total_steps'] - 1
            self.config.start_save_epoch = rt_config['total_epochs'] - 1
        if rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch:
            assert len(logits.shape) == 2
            predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs["_index"])

            save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}.csv")
            sentences = []
            predicts = []
            ground_truths = []
            for predict_index, index, label_id in zip(predict_indexes, indexes, label_ids):
                sentence = origin_data.iloc[int(index)][self.config.sentence]
                predict = self.label_vocab.get_word(predict_index)
                ground_truth = self.label_vocab.get_word(label_id)
                sentences.append(sentence)
                predicts.append(predict)
                ground_truths.append(ground_truth)

            logger.info(f"Save the {stage} predict data at {save_file}")
            save_data = pd.DataFrame({"sentence": sentences, "predict": predicts, "ground truth": ground_truths}, index=indexes)
            save_data.to_csv(save_file)
            
        if 'loss' in outputs:
            log_info[self.loss_name_map(stage)] = torch.sum(outputs['loss'])/outputs['loss'].size(0)
        return log_info
