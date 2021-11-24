import hjson
import pickle as pkl
import pandas as pd
from typing import Union, Dict
from dlkit.utils.parser import BaseConfigParser, PostProcessorConfigParser
from dlkit.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlkit.utils.config import ConfigTool
import torch


        

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
                "logits": "logits"
            },
            "origin_data": {
                "sentence": "sentence"
            }
        }
    }
    """

    def __init__(self, config: Dict):
        super(ClassificationPostProcessorConfig, self).__init__(config)

        self.logits = self.output_data['logits']
        self.sentence = self.config['origin_data']['sentence']
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


@postprocessor_register('classification')
class ClassificationPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: ClassificationPostProcessorConfig):
        super(ClassificationPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab

    def process(self, stage, outputs, origin_data)->Dict:
        """TODO: Docstring for process.
        :data: TODO
        :returns: TODO
        """
        print(outputs)
        # for output in outputs:
            # print(output)
            # logits = output[self.config.logits]
            # index = output['_index']
            # sentence = origin_data.iloc[index][self.config.sentence]
            # print("one example")
            # print('logits')
            # print(logits)
            # print('sentence')
            # print(sentence)
            # print("one example")

        raise PermissionError
        if 'loss' in outputs:
            return {self.loss_name_map(stage): torch.sum(outputs['loss'])}
        return {}
