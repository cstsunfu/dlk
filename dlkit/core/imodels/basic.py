import hjson
from typing import Dict, Union, Callable, List
from dlkit.core.models import model_register, model_config_register
from dlkit.core.optimizers import optimizer_register, optimizer_config_register
from dlkit.core.schedules import schedule_register, schedule_config_register
from dlkit.core.losses import loss_register, loss_config_register
from dlkit.utils.config import ConfigTool
from . import imodel_config_register, imodel_register, GatherOutputMixin

import pytorch_lightning as pl

@imodel_config_register('basic')
class BasicIModelConfig(object):
    """docstring for IModelConfig"""
    def __init__(self, config):
        super(BasicIModelConfig, self).__init__()
        self.model, self.model_config = self.get_model(config.get("model"))
        self.loss, self.loss_config = self.get_loss(config.get("loss"))

        self.optimizer, self.optimizer_config = self.get_optimizer(config.get("optimizer", 'basic'))

        self.schedule, self.schedule_config = self.get_schedule(config.get("schedule", "basic"))

        self.config = config.pop('config', {})

    def get_model(self, config):
        """get embedding config and embedding module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(model_register, model_config_register, "model", config)

    def get_loss(self, config):
        """get encoder config and encoder module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(loss_register, loss_config_register, "loss", config)
        
    def get_optimizer(self, config):
        """get optimizer

        :config: TODO
        :returns: TODO
        """
        return ConfigTool.get_leaf_module(optimizer_register, optimizer_config_register, "optimizer", config)

    def get_schedule(self, config):
        """get decoder config and decoder module

        :config: TODO
        :returns: TODO
        """
        return ConfigTool.get_leaf_module(schedule_register, schedule_config_register, "schedule", config)


@imodel_register("basic")
class BasicIModel(pl.LightningModule, GatherOutputMixin):
    """
    """
    def __init__(self, config: BasicIModelConfig):
        super().__init__()
        self.config = config  # schedule will init in configure_optimizers, because it will use the datamodule info

        self.model = config.model(config.model_config)


        self.calc_loss = config.loss(config.loss_config)
        self.get_optimizer = config.optimizer(model=self.model, config=config.optimizer_config)

        self._origin_validation_data = None
        self._origin_test_data = None

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = train_loader.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def postprocessor(self, outputs):
        """you can overwrite this part to post process the outputs of validation_epoch_end or test_epoch_end

        :outputs: TODO
        :returns: TODO

        """
        return outputs

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        result = self.model.training_step(batch)
        loss = self.calc_loss(result, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.model.validation_step(batch)
        loss = self.calc_loss(result, batch)
        return {"loss": loss, "index": batch["_index"], "predict": result}

    def validation_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        outputs = self.gather_outputs(outputs)

        if self.local_rank in [0, -1]:
            key_all_ins_map = self.concat_list_of_dict_outputs(outputs)
            self.postprocessor(key_all_ins_map)
        return outputs

    def test_step(self, batch, batch_idx):
        result = self.model.test_step(batch)
        loss = self.calc_loss(result, batch)
        return {"loss": loss, "index": batch["_index"], "predict": result}

    def test_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        outputs = self.gather_outputs(outputs)

        if self.local_rank in [0, -1]:
            key_all_ins_map = self.concat_list_of_dict_outputs(outputs)
            self.postprocessor(key_all_ins_map)
        return outputs

    def predict_step(self, batch, batch_idx):
        return self.model.predict_step(batch)

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        schedule_config = self.config.schedule_config.schedule_config
        num_warmup_steps = schedule_config.get('num_warmup_steps', 0)
        if num_warmup_steps>0 and num_warmup_steps<1:
            schedule_config["num_warmup_steps"] = self.total_steps * num_warmup_steps
        schedule_config["num_training_steps"] = self.total_steps

        self.config.schedule.schedule_config.schedule_config = schedule_config
        schedule = self.config.schedule(optimizer, self.config.schedule_config)

        return { 
            "optimizer": optimizer,
            "schedule": schedule 
        }
