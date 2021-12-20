from dlk.utils.parser import config_parser_register
import json
import hjson
from typing import Dict, Union, Any
from dlk.data.processors import processor_config_register, processor_register


class Processor(object):
    """docstring for Processor"""
    def __init__(self, config: Union[str, Dict]):
        super(Processor, self).__init__()
        self.config = config
        if not isinstance(config, dict):
            config = hjson.load(open(config), object_pairs_hook=dict)
            config = config_parser_register.get("processor")(config).parser_with_check()
            assert len(config) == 1, f"Currently we didn't support search for Processor, if you require this feature please create an issue to describe the reason details."
            self.config = config[0]
        
    def fit(self, data: Dict[str, Any], stage='train'):
        """TODO: Docstring for fit.
        :returns: TODO

        """
        processor = processor_register.get(self.config.get('_name'))(stage=stage, config=processor_config_register.get(self.config.get('_name'))(stage=stage, config=self.config))
        return processor.process(data)
