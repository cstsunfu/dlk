# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict

from datasets import Dataset
from intc import MISSING, Base, StrField, cregister

from dlk.data.subprocessor.fast_tokenizer import FastTokenizer, FastTokenizerConfig
from dlk.data.subprocessor.token2id import Token2ID, Token2IDConfig
from dlk.data.subprocessor.token_gather import TokenGather, TokenGatherConfig
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("processor", "my")
class DefaultProcessorConfig(Base):
    """the default processor"""

    processed_data_dir = StrField(
        value="data/processed_data",
        help="the save dir of the processor, not effective by `data_root`.",
    )
    meta_dir = StrField(
        value="data/meta_data",
        help="the save dir of the meta info, not effective by `data_root`.",
    )
    tokenizer_path = StrField(
        value=MISSING,
        help="the config path for the tokenizer, this is not effected by `meta_dir`",
    )


@register("processor", "my")
class DefaultProcessor(object):
    """
    Custom Processor Example.
    Demonstrates how to manually orchestrate subprocessors and save data using the new HF Datasets backend.
    """

    stage_data_set_map = {
        "train": "train_data_set",
        "online": "online_data_set",
    }

    def __init__(self, stage: str, config: DefaultProcessorConfig):
        super(DefaultProcessor, self).__init__()
        self.stage = stage
        assert (
            stage in self.stage_data_set_map
        ), f"stage {stage} not supported for this example"
        self.config: DefaultProcessorConfig = config

        # Initialize Subprocessors
        self.tokenizer = FastTokenizer(
            stage=stage,
            config=FastTokenizerConfig._from_dict(
                {"tokenizer_path": config.tokenizer_path}
            ),
            meta_dir=config.meta_dir,
        )
        self.label_gather = TokenGather(
            stage=stage,
            config=TokenGatherConfig._from_dict(
                {
                    "gather_columns": ["labels"],
                    "token_vocab": "label_vocab.json",
                    "unk": "",
                    "pad": "",
                }
            ),
            meta_dir=config.meta_dir,
        )
        self.label2id = Token2ID(
            stage=stage,
            config=Token2IDConfig._from_dict(
                {
                    "input_map": {"tokens": "labels"},
                    "output_map": {"token_ids": "label_ids"},
                    "vocab": "label_vocab.json",
                }
            ),
            meta_dir=config.meta_dir,
        )

    def save(self, data: Dataset, type_name: str):
        """Save data to disk using HF Datasets (Arrow format).

        Args:
            data: processed Dataset
            type_name: 'train' or 'valid'
        """
        save_path = os.path.join(self.config.processed_data_dir, type_name)
        os.makedirs(save_path, exist_ok=True)

        data.save_to_disk(save_path)
        logger.info(f"Saved {type_name} data to {save_path}")

    def process(self, data: Dict) -> Dict:
        """Process entry for train stage

        Args:
            data: {"train": Dataset, "valid": Dataset}

        Returns:
            processed data dict
        """
        train_data: Dataset = data["train"]
        valid_data: Dataset = data["valid"]

        # 1. Tokenize
        train_data = self.tokenizer.process(train_data, deliver_meta=False)
        valid_data = self.tokenizer.process(valid_data, deliver_meta=False)

        # 2. Gather Labels (Only on train)
        self.label_gather.process(train_data, deliver_meta=True)

        # 3. Map Labels to IDs
        train_data = self.label2id.process(train_data, deliver_meta=False)
        valid_data = self.label2id.process(valid_data, deliver_meta=False)

        # 4. Save to Disk
        self.save(train_data, "train")
        self.save(valid_data, "valid")

        return {}

    def online_process(self, data: Dict):
        """Online server process without save."""
        # For dictionary dict of lists
        if "labels" not in data:
            data["label_ids"] = [[0]] * len(next(iter(data.values())))
        # Tokenize
        return self.tokenizer.process_batch(data, deliver_meta=False)
