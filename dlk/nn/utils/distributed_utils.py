# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gathers tensors from all process and supports backward propagation.

    This is essential for contrastive learning to scale the effective batch size
    across multiple GPUs, maximizing the negative sample pool.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> tuple:
        """Forward pass to gather tensors from all ranks.

        Args:
            ctx: Context object to store information for backward.
            x (torch.Tensor): Input tensor on the local GPU.

        Returns:
            tuple: A tuple of tensors gathered from all GPUs.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return (x,)

        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        """Backward pass to route gradients to the correct GPU.

        Args:
            ctx: Context object.
            *grads (torch.Tensor): Gradients corresponding to the gathered tensors.

        Returns:
            torch.Tensor: Gradient for the local input tensor.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return grads[0]

        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_tensor_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Helper function to gather tensor across GPUs while keeping computation graph.

    Args:
        tensor (torch.Tensor): The local tensor to be gathered.

    Returns:
        torch.Tensor: A concatenated tensor from all GPUs along dimension 0.
    """
    if dist.is_available() and dist.is_initialized():
        gathered = GatherLayer.apply(tensor)
        return torch.cat(gathered, dim=0)
    return tensor
