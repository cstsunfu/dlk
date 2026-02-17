# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Callable, Dict, List, Type, Union

import pandas as pd
from datasets import Dataset
from intc import (
    MISSING,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    dataclass,
)

from dlk.utils.import_module import import_module_dir

logger = logging.getLogger(__name__)


@dataclass
class BaseSubProcessorConfig(Base):
    """The base configuration for all subprocessors."""

    collect_data_set = ListField(
        value=[],
        suggestions=[["train", "valid", "test"]],
        help="Datasets processed during collect stage.",
    )
    train_data_set = ListField(
        value=[],
        suggestions=[["train", "valid", "test"]],
        help="Datasets processed during train stage.",
    )
    predict_data_set = ListField(
        value=[],
        suggestions=[["predict"], []],
        help="Datasets processed during predict stage.",
    )
    online_data_set = ListField(
        value=[],
        suggestions=[["online"], []],
        help="Datasets processed during online stage.",
    )
    input_map = DictField(
        value={}, help="Mapping of required input keys to provided dataset keys."
    )
    output_map = DictField(
        value={}, help="Mapping of processor outputs to dataset keys."
    )


class BaseSubProcessor(object):
    """Base class for all subprocessors.

    Attributes:
        PROCESSOR_MODE (str): Defines the execution paradigm of the processor.
            - "map": Stateless transformation. Safe for multi-processing via `datasets.map`.
            - "gather": Stateful accumulation (e.g., building vocabularies).
                        Must be executed sequentially in the main process to preserve global state.
    """

    PROCESSOR_MODE = "map"

    def __init__(self, stage: str, config: BaseSubProcessorConfig, meta_dir: str):
        """Initializes the BaseSubProcessor.

        Args:
            stage (str): The current pipeline stage (e.g., 'train', 'predict').
            config (BaseSubProcessorConfig): Configuration object.
            meta_dir (str): Directory path to save/load meta information like vocabularies.
        """
        self.stage = stage
        self.config = config
        self.meta_dir = meta_dir
        self.loaded_meta = False

    def load_meta(self):
        """Loads meta information required for processing (e.g., vocabularies)."""
        self.loaded_meta = True

    def save_meta(self):
        """Saves accumulated global state to disk. Used primarily in 'gather' mode."""
        pass

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Processes a batch of data. Must be implemented by subclasses.

        Args:
            batch (Dict[str, List[Any]]): A dictionary of lists representing column data.
            deliver_meta (bool): Whether to deliver/save meta info (legacy argument, keeping for compatibility).

        Returns:
            Dict[str, List[Any]]: The transformed batch data.
        """
        raise NotImplementedError(
            f"SubProcessor {self.__class__.__name__} must implement `process_batch`."
        )

    def process(self, data: Any, deliver_meta: bool, **kwargs) -> Any:
        """Entry point for processing data. Dynamically routes based on PROCESSOR_MODE.

        Args:
            data (Any): Input data (`datasets.Dataset`, `dict`, or `pd.DataFrame`).
            deliver_meta (bool): Whether to finalize and save global states.
            **kwargs: Additional args, like `num_proc` for multi-processing.

        Returns:
            Any: The processed data in its original type.
        """
        assert isinstance(
            data, Dataset
        ), f"Unsupported data type: {type(data)}. Only support huggingface dataset"
        if self.PROCESSOR_MODE == "map":
            # Stateless map: highly efficient, multi-process, cached
            return data.map(
                lambda b: self.process_batch(b, deliver_meta=deliver_meta),
                batched=True,
                num_proc=kwargs.get("num_proc", os.cpu_count()),
                desc=f"Map: {self.__class__.__name__}",
            )
        else:
            assert self.PROCESSOR_MODE == "gather"
            logger.info(
                f"Gathering state sequentially for {self.__class__.__name__}..."
            )
            for i in range(0, len(data), 1000):
                batch = data[i : i + 1000]
                self.process_batch(batch, deliver_meta=False)

            if deliver_meta:
                self.save_meta()
            return data  # Gather mode does not modify the data itself


subprocessor_dir = os.path.dirname(__file__)
import_module_dir(subprocessor_dir, "dlk.data.subprocessor")
