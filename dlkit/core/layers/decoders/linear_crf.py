import torch
from typing import Dict, List, Set
from dlkit.core.base_module import BaseModule
from . import decoder_register, decoder_config_register
from dlkit.core.modules import module_config_register, module_register

@decoder_config_register("linear_crf")
class LinearCRFConfig(object):
    """docstring for LinearCRFConfig
    {
        module@linear: {
            _base: linear,
        },
        module@crf: {
            _base: crf,
        },
        config: {
            input_size: "*@*",
            output_size: "*@*",
            return_logits: "decoder_logits",
            reduction: "mean",
            output_map: {}
        },
        _link:{
            config.input_size: [module@linear.config.input_size],
            config.output_size: [module@linear.config.output_size, module@crf.config.output_size],
            config.reduction: [module@crf.config.reduction],
        }
        _name: "linear_crf",
    }
    """
    def __init__(self, config: Dict):
        super(LinearCRFConfig, self).__init__()
        self.linear_config = config["module@linear"]
        self.crf_config = config["module@crf"]
        self.return_logits = config['config']['return_logits']
        self.output_map = config['config']['output_map']
        

@decoder_register("linear_crf")
class LinearCRF(BaseModule):
    def __init__(self, config: LinearCRFConfig):
        super(LinearCRF, self).__init__()
        self._provide_keys = {'logits'}
        self._required_keys = {'embedding', 'label_ids', 'attention_mask'}
        self._provided_keys = set()

        self.config = config
        self.linear = module_register.get('linear')(module_config_register.get('linear')(config.linear_config))
        self.crf = module_register.get('crf')(module_config_register.get('crf')(config.crf_config))

    def provide_keys(self)->Set:
        """TODO: should provide_keys in model?
        """
        return self.set_rename(self._provided_keys.union(self._provide_keys), self.config.output_map)

    def check_keys_are_provided(self, provide: Set[str])->None:
        """
        """
        self._provided_keys = provide
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The {self.__class__.__name__} Module required '{required_key}' as input.")

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """

        raise NotImplementedError

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """TODO: Docstring for training_step.
        :arg1: TODO
        :returns: TODO

        """
        logits = self.linear(inputs['embedding'])
        loss = self.crf.training_step(logits, inputs['label_ids'], inputs['attention_mask'])
        if self.config.return_logits:
            inputs[self.config.return_logits] = logits
        inputs['loss'] = loss
        return self.dict_rename(inputs, self.config.output_map)

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """TODO: Docstring for training_step.

        :arg1: TODO
        :returns: TODO

        """
        logits = self.linear(inputs['embedding'])
        loss = self.crf.training_step(logits, inputs['label_ids'], inputs['attention_mask'])
        if self.config.return_logits:
            inputs[self.config.return_logits] = logits
        inputs['loss'] = loss


        return self.dict_rename(inputs, self.config.output_map)
        return {}

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        return self.predict_step(inputs)
