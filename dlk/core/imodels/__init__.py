"""imodels"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register
import torch

imodel_config_register = Register("IModel config register.")
imodel_register = Register("IModel register.")

class GatherOutputMixin(object):
    """gather all the small batches output to a big batch"""


    @staticmethod
    def proc_dist_outputs(dist_outputs):
        """gather all distributed outputs to outputs which is like in a single worker.

        :dist_outputs: the inputs of pytorch_lightning *_epoch_end when using ddp
        :returns: the inputs of pytorch_lightning *_epoch_end when only run on one worker.
        """
        outputs = []
        for dist_output in dist_outputs:
            one_output = {}
            for key in dist_output:
                try:
                    one_output[key] = torch.cat(torch.swapaxes(dist_output[key], 0, 1).unbind(), dim=0)
                except:
                    raise KeyError(f"{key}: {dist_output[key]}")
            outputs.append(one_output)
        return outputs

    def gather_outputs(self, outputs):
        """TODO: Docstring for gather_outputs.

        :outputs: TODO
        :returns: TODO

        """
        if self.trainer.world_size>1:
            dist_outputs = self.all_gather(outputs)
            if self.local_rank in [0, -1]:
                outputs = self.proc_dist_outputs(dist_outputs)
        return outputs

    def concat_list_of_dict_outputs(self, outputs):
        """only support all the outputs has the same dim
        :returns: TODO
        """
        key_all_batch_map = {}
        for batch in outputs:
            for key in batch:
                if key not in key_all_batch_map:
                    key_all_batch_map[key] = []
                key_all_batch_map[key].append(batch[key])
        key_all_ins_map = {}
        for key in key_all_batch_map:
            key_all_ins_map[key] = torch.cat(key_all_batch_map[key], dim=0)

        return key_all_ins_map


def import_imodels(imodels_dir, namespace):
    for file in os.listdir(imodels_dir):
        path = os.path.join(imodels_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            imodel_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + imodel_name)


# automatically import any Python files in the imodels directory
imodels_dir = os.path.dirname(__file__)
import_imodels(imodels_dir, "dlk.core.imodels")
