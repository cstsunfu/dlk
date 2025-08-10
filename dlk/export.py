# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    EnumField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
    dataclass,
)

from dlk.utils.register import register, register_module_name

try:
    import onnx
    import onnxruntime
except ImportError:
    print(
        "Warning: 'onnx' or 'onnxruntime' not found. "
        "Please install them with `pip install onnx onnxruntime` "
        "to use ONNX export and validation functionalities."
    )

from dlk.predict import Predict
from dlk.preprocess import PreProcessor
from dlk.utils.io import open
from dlk.utils.onnx_export_wrap import GenericOnnxExportWrapper

logger = logging.getLogger(__name__)


@cregister("export")
class ExportConfig(Base):
    """the base loss config"""

    input_names = ListField(
        value=["input_ids", "type_ids", "attention_mask", "_index"],
        help="List of input tensor names for the ONNX model.",
    )
    output_names = ListField(
        value=["logits", "head_logits", "_index"],
        help="List of output tensor names for the ONNX model.",
    )
    process_config = StrField(
        value="./config/processor.jsonc",
        help="Path to the preprocessing configuration file.",
    )
    fit_config = StrField(
        value="./config/fit.jsonc",
        help="Path to the training/fitting configuration file.",
    )
    checkpoint = StrField(
        value="./logs/0/checkpoint/last.ckpt",
        help="Path to the model checkpoint (.ckpt) file.",
    )
    output_path = StrField(
        value="model.onnx",
        help="Path where the exported ONNX model will be saved.",
    )
    opset_version = IntField(
        value=14,
        help="ONNX opset version to use for the export. Default is 14.",
    )
    dynamic_axes = DictField(
        value={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "type_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "_index": {0: "batch_size"},
            # It's good practice to also define dynamic axes for outputs
            "logits": {0: "batch_size"},
            "head_logits": {0: "batch_size"},
        },
        help="Dictionary specifying dynamic axes for inputs/outputs. "
        "Example: {'input_ids': {0: 'batch', 1: 'sequence'}}.",
    )
    validate = BoolField(
        value=True,
        help="If True, validates the exported ONNX model against the original PyTorch model.",
    )


class Export:
    """Orchestrates the process of exporting a PyTorch model to ONNX format."""

    def __init__(self, config: ExportConfig):
        """Initializes the Export object.

        Args:
            input_names: A list of input tensor names for the ONNX model.
            output_names: A list of output tensor names for the ONNX model.
            process_config: Path to the preprocessing configuration file.
            fit_config: Path to the training/fitting configuration file.
            checkpoint: Path to the model checkpoint (.ckpt) file.
            output_path: The path where the exported ONNX model will be saved.
            opset_version: The ONNX opset version to use for the export.
            dynamic_axes: A dictionary specifying dynamic axes for inputs/outputs.
                Example: `{'input_ids': {0: 'batch', 1: 'sequence'}}`.
            validate: If True, validates the exported ONNX model against the
                original PyTorch model.
        """
        super(Export, self).__init__()
        self.processor = PreProcessor(config.process_config, stage="online")
        self.output_path = config.output_path
        self.opset_version = config.opset_version
        self.input_names = config.input_names
        self.output_names = config.output_names
        self.dynamic_axes = config.dynamic_axes or {}
        self.validate = config.validate

        # The Predict class is used here to conveniently load the model
        # and its associated data module from configuration and a checkpoint.
        predict = Predict(config.fit_config, config.checkpoint)
        datamodule, _ = predict.get_datamodule(
            predict.dlk_config,
            {},
            world_size=predict.trainer.world_size,
        )
        self.datamodule = datamodule
        self.model = predict.imodel.model
        self.model.eval()  # Set the model to evaluation mode

    @staticmethod
    def _prepare_for_export(
        model: nn.Module,
        input_names: List[str],
        output_names: List[str],
        dummy_input_batch: Dict[str, torch.Tensor],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Tuple[GenericOnnxExportWrapper, Tuple[torch.Tensor, ...], Dict[str, Any]]:
        """Prepares all necessary components for ONNX export.

        Args:
            model: The original PyTorch model.
            input_names: The names and order of the inputs.
            output_names: The names and order of the outputs.
            dummy_input_batch: A dictionary of tensors to be used for tracing.
                Keys are input names, values are torch.Tensors.
            dynamic_axes: A dictionary describing dynamic axes.

        Returns:
            A tuple containing:
                - The wrapped model ready for export.
                - A tuple of dummy inputs for tracing.
                - A dictionary of keyword arguments for `torch.onnx.export`.
        """
        # 1. Create the generic wrapper instance.
        exporter = GenericOnnxExportWrapper(model, input_names, output_names)
        exporter.eval()

        # 2. Create the dummy inputs tuple, ensuring the order matches input_names.
        dummy_inputs_tuple = tuple(dummy_input_batch[name] for name in input_names)

        # 3. Prepare the keyword arguments for torch.onnx.export.
        export_kwargs = {
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes if dynamic_axes else None,
        }

        return exporter, dummy_inputs_tuple, export_kwargs

    def export(self, input_df: pd.DataFrame):
        """Processes input data and exports the model to ONNX.

        Args:
            input_df: A pandas DataFrame containing sample data to generate a
                dummy input for tracing the model graph.
        """
        # Generate a single batch of data to be used for tracing the model.
        # Note: `processor.fit` here is likely used for transformation, not training.
        processed_data = self.processor.fit(input_df)
        batch = self.datamodule.online_process_batch(processed_data)

        exporter, dummy_inputs_tuple, export_kwargs = self._prepare_for_export(
            model=self.model,
            input_names=self.input_names,
            output_names=self.output_names,
            dummy_input_batch=batch,
            dynamic_axes=self.dynamic_axes,
        )

        logger.info(f"Starting ONNX export to {self.output_path}...")
        torch.onnx.export(
            exporter,
            dummy_inputs_tuple,
            self.output_path,
            opset_version=self.opset_version,
            **export_kwargs,
        )
        logger.info("ONNX export completed successfully.")

        if self.validate:
            verify_onnx_model(
                onnx_path=self.output_path,
                dummy_inputs_tuple=dummy_inputs_tuple,
                pytorch_model=exporter,
            )


def verify_onnx_model(
    onnx_path: str,
    dummy_inputs_tuple: Tuple[torch.Tensor, ...],
    pytorch_model: nn.Module,
):
    """Verifies the ONNX model by comparing its output with the PyTorch model.

    Args:
        onnx_path: Path to the saved ONNX model file.
        dummy_inputs_tuple: The same tuple of dummy inputs used for export.
        pytorch_model: The PyTorch model (or wrapper) that was exported.
    """
    logger.info("\nVerifying the exported ONNX model...")

    # 1. Check if the ONNX model is well-formed.
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model format check passed.")

    # 2. Create an ONNX Runtime inference session.
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # 3. Prepare inputs for ONNX Runtime.
    # The keys must match the input names provided during export.
    try:
        ort_inputs = {
            ort_session.get_inputs()[i].name: to_numpy(dummy_inputs_tuple[i])
            for i in range(len(dummy_inputs_tuple))
        }
    except Exception as e:
        raise RuntimeError(
            "Failed to prepare inputs for ONNX Runtime. "
            "Ensure the input names match those used during export."
            f"ONNX input names: {[input.name for input in ort_session.get_inputs()]}"
        ) from e

    # 4. Run inference with ONNX Runtime.
    logger.info("Running inference with ONNX Runtime...")
    ort_outputs = ort_session.run(None, ort_inputs)

    # 5. Run inference with the PyTorch model to get a baseline.
    logger.info("Running inference with PyTorch to get baseline results...")
    with torch.no_grad():
        pytorch_outputs = pytorch_model(*dummy_inputs_tuple)

    # 6. Compare the outputs from both models.
    logger.info("Comparing ONNX Runtime and PyTorch outputs...")
    if len(ort_outputs) != len(pytorch_outputs):
        raise RuntimeError(
            f"Mismatch in number of outputs. PyTorch: {len(pytorch_outputs)}, "
            f"ONNX: {len(ort_outputs)}"
        )

    for i, (ort_out, pt_out) in enumerate(zip(ort_outputs, pytorch_outputs)):
        np.testing.assert_allclose(
            to_numpy(pt_out),
            ort_out,
            rtol=1e-03,
            atol=1e-05,
            err_msg=f"Output {i} mismatch",
        )
        logger.info(f"Output {i} passed verification.")

    logger.info("Validation successful! ONNX model outputs match PyTorch outputs.")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array."""
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def main():
    """Main function to run the ONNX export process."""
    # Load sample data for tracing the model.
    # Only a small amount is needed.
    with open("./test.json", "r") as f:
        data = json.load(f)
    data = data[:10]
    data = [{"sentence": item["sentence"], "uuid": item["uuid"]} for item in data]
    input_df = pd.DataFrame(data)

    config = ExportConfig()

    # Define the configuration for the export process.
    export_task = Export(config=config)

    # Run the export process.
    export_task.export(input_df)


if __name__ == "__main__":
    main()
