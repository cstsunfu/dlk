import enum
from typing import Dict
import torch.nn as nn
from . import loss_register, loss_config_register
import torch.nn as nn
import copy

TASK_PRED_TRUTH_PAIRS = {
    "classification": [
        ["embedding", "label_id"]
    ],
    "mrc": [
        ["start_logits", "start_position"],
        ["end_logits", "end_position"]
    ]
}

@loss_config_register("cross_entropy")
class CrossEntropyLossConfig(object):
    """docstring for CrossEntropyLossConfig
    {
        config: {
            task_name: "classification",
            weight: null, # or a list of value for every class
            ignore_index: -1,
            label_smoothing: 0.0, # torch>=1.10
            pred_truth_pair: [], # overwrite the TASK_PRED_TRUTH_PAIRS
            loss_scale: [], # scale the loss, if loss_scale 
        },
        _name: "cross_entropy",
    }
    """
    def __init__(self, config: Dict):
        super(CrossEntropyLossConfig, self).__init__()
        config = config.get('config', {})
        self.task_name = config.get('task_name', "") # must provide
        self.weight = config.get('weight', None)
        self.ignore_index = config.get('ignore_index', -1)
        self.label_smoothing = config.get('label_smoothing', 0.0)
        self.pred_truth_pair = config.get('pred_truth_pair', [])
        if self.task_name not in TASK_PRED_TRUTH_PAIRS and not self.pred_truth_pair:
            raise PermissionError(f"The cross_entropy loss is not defined for {self.task_name}.")
        if not self.pred_truth_pair:
            self.pred_truth_pair = TASK_PRED_TRUTH_PAIRS.get(self.task_name, [])
        self.loss_scale = config.get("config", [])
        if self.loss_scale:
            assert len(self.loss_scale) == len(self.pred_truth_pair), f"loss scale number must equals to loss num"
        else:
            self.loss_scale = [1]*len(self.pred_truth_pair) if len(self.pred_truth_pair)>0 else []


@loss_register("cross_entropy")
class CrossEntropyLoss(object):
    def __init__(self, config: CrossEntropyLossConfig):
        super(CrossEntropyLoss, self).__init__()
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=config.weight,
            ignore_index=config.ignore_index, 
            label_smoothing=config.label_smoothing 
        )

    def calc(self, result, inputs):
        """TODO: Docstring for get_loss.
        :returns: TODO
        """

        loss = 0
        for i, (pred, truth) in enumerate(self.config.pred_truth_pair):
            loss = loss + self.cross_entropy(result[pred], inputs[truth]) * self.config.loss_scale[i]
        return loss

    def __call__(self, result, inputs):
        """TODO: Docstring for __call__.
        :returns: TODO

        """
        return self.calc(result, inputs)
