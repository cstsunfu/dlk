# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple

import torch
from torch import nn


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
