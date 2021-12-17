import sys
sys.path.append('../')
import pickle as pkl

# from dlk.utils.vocab import Vocabulary
# from dlk.models import DECODER_CONFIG_REGISTRY

# a = Vocabulary()
# a.auto_update(['ni', 'hao'])

# pkl.dump(a, open('vocab.pkl', 'wb'))

a = pkl.load(open('./vocab.pkl', 'rb'))
print(a.get_index('hao'))
