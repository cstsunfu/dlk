from dlkit.utils.logger import setting_logger
import hjson
setting_logger("")
from dlkit.utils.parser import config_parser_register
import json

path = './examples/sequence_labeling/conll2003/norm_char_lstm_crf/output/textadamtask=crf_lstm_weight_decay=0.01_batch_size=10_lstm_output_size=200/config.json'

parser = config_parser_register.get('root')

config = hjson.load(open(path, 'r'))
# config.pop('_focus')
config = parser(config).parser()
print(json.dumps(config, indent=4))
