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

logger = logging.getLogger(__name__)


class GenericOnnxExportWrapper(nn.Module):
    """A generic wrapper to export PyTorch models that use dictionaries.

    This wrapper converts the tuple-based inputs (*args) expected by ONNX
    into the dictionary-based input required by the model. It then converts
    the model's dictionary-based output back into a tuple for ONNX.
    """

    def __init__(
        self, model: nn.Module, input_names: List[str], output_names: List[str]
    ):
        """Initializes the GenericOnnxExportWrapper.

        Args:
            model: The original PyTorch model (nn.Module) that accepts and
                returns dictionaries.
            input_names: A list of strings defining the names and order of the
                input tensors. This order must strictly match the order of the
                `dummy_inputs` tuple provided to `torch.onnx.export`.
            output_names: A list of strings defining the names and order of the
                tensors to be extracted from the model's output dictionary.
        """
        super().__init__()
        if not hasattr(model, "forward") or not callable(model.forward):
            raise TypeError(
                "The provided 'model' must be a valid nn.Module with a "
                "callable forward method."
            )
        if not isinstance(input_names, list) or not isinstance(output_names, list):
            raise TypeError(
                "'input_names' and 'output_names' must be lists of strings."
            )

        self.model = model
        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Defines the ONNX-friendly forward pass.

        Args:
            *args: A tuple of input tensors.

        Returns:
            A tuple of output tensors.
        """
        # 1. Convert the input tuple (*args) to a dictionary for the model.
        # The zip function ensures the order is preserved from `input_names`.
        inputs_dict = {name: tensor for name, tensor in zip(self.input_names, args)}

        # 2. Call the original model with the dictionary of inputs.
        outputs_dict = self.model(inputs=inputs_dict)

        # 3. Extract tensors from the output dictionary in the specified order.
        # This ensures the output tuple has a predictable and fixed order.
        outputs_tuple = tuple(outputs_dict[name] for name in self.output_names)

        return outputs_tuple


class Export:
    """Orchestrates the process of exporting a PyTorch model to ONNX format."""

    def __init__(
        self,
        input_names: List[str],
        output_names: List[str],
        process_config: str,
        fit_config: str,
        checkpoint: str,
        output_path: str = "model.onnx",
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        validate: bool = True,
    ):
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
        self.processor = PreProcessor(process_config, stage="online")
        self.output_path = output_path
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes or {}
        self.validate = validate

        # The Predict class is used here to conveniently load the model
        # and its associated data module from configuration and a checkpoint.
        predict = Predict(fit_config, checkpoint)
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

    # Define the configuration for the export process.
    export_task = Export(
        input_names=[
            "input_ids",
            "type_ids",
            "attention_mask",
            "_index",
        ],
        # NOTE: Removed duplicate "head_logits" from the original list.
        output_names=[
            "logits",
            "head_logits",
            "_index",
        ],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "type_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            # "special_tokens_mask": {0: "batch_size", 1: "sequence_length"},
            "_index": {0: "batch_size"},
            # It's good practice to also define dynamic axes for outputs
            "logits": {0: "batch_size"},
            "head_logits": {0: "batch_size"},
            "_index": {0: "batch_size"},
        },
        process_config="./config/processor.jsonc",
        fit_config="./config/fit.jsonc",
        checkpoint="./logs/0/checkpoint/last.ckpt",
        output_path="model.onnx",
        validate=True,
    )

    # Run the export process.
    export_task.export(input_df)


if __name__ == "__main__":
    main()
