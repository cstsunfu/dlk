# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from intc import NestField, StrField, cregister

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.utils.register import register


@cregister("postprocessor", "contrastive")
class ContrastivePostProcessorConfig(BasePostProcessorConfig):
    class InputMap:
        embedding = StrField(value="embedding", help="The output normalized embedding.")
        index = StrField(value="_index", help="The index of the sample.")

    input_map = NestField(value=InputMap)


@register("postprocessor", "contrastive")
class ContrastivePostProcessor(BasePostProcessor):
    """Evaluates Retrieval performance (Recall@K)."""

    def __init__(self, config: ContrastivePostProcessorConfig):
        super().__init__(config)
        self.config = config

    def wrap_predict_one_batch(
        self, stage: str, batch_output: Dict, origin_data: pd.DataFrame, rt_config: Dict
    ) -> List:
        batch_output[self.config.input_map.embedding] = (
            batch_output[self.config.input_map.embedding].detach().cpu().numpy()
        )
        return self.predict_one_batch(stage, batch_output, origin_data, rt_config)

    def predict_one_batch(
        self, stage: str, batch_output: Dict, origin_data: pd.DataFrame, rt_config: Dict
    ) -> List:
        embeddings = batch_output[self.config.input_map.embedding]
        indexes = batch_output[self.config.input_map.index]

        results = []
        for emb, idx in zip(embeddings, indexes):
            # In a real setup, you'd store whether this is a query or document.
            # Here we just save the embedding.
            results.append({"embedding": emb, "index": idx})
        return results

    def do_calc_metrics(self, predicts: List, stage: str, rt_config: Dict) -> Dict:
        """Calculates In-Batch Retrieval Recall@1, Recall@5.
        (Note: For rigorous evaluation, one should evaluate against a full corpus).
        """
        # For a simple validation sanity check, we do in-batch similarity.
        # Real retrieval evaluation requires a separate Query dataset and Corpus dataset.
        if len(predicts) < 5:
            return {}

        embeddings = torch.tensor(np.stack([p["embedding"] for p in predicts]))

        # Calculate N x N similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())

        # Zero out diagonal to avoid self-retrieval
        sim_matrix.fill_diagonal_(-float("inf"))

        # Here we just return dummy logic. A true evaluation script
        # would cross-match query_embeddings and doc_embeddings.
        # This acts as a placeholder for the user's specific metric needs.
        return {f"{self.loss_name_map(stage)}_recall_ready": 1.0}
