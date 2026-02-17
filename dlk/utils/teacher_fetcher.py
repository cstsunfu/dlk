# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import numpy as np
import requests
from intc import Base, DictField, IntField, ListField, StrField, cregister, dataclass

from dlk.utils.import_module import import_module_dir
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@dataclass
class BaseTeacherFetcherConfig(Base):
    """The base configuration for teacher fetcher."""

    timeout = IntField(
        value=10, help="Timeout for fetching teacher results in seconds."
    )


class BaseTeacherFetcher:
    """Base class for all teacher fetchers."""

    def __init__(self, config: BaseTeacherFetcherConfig):
        self.config = config

    def fetch(self, request_data: List[Dict[str, Any]]) -> List[Any]:
        """Fetch teacher predictions for a batch of data.

        Args:
            request_data: A list of dictionaries representing the batch inputs.

        Returns:
            A list of predictions (e.g., logits) from the teacher model.
        """
        raise NotImplementedError


@cregister("teacher_fetcher", "http")
class HttpTeacherFetcherConfig(BaseTeacherFetcherConfig):
    """Configuration for HTTP-based teacher fetcher."""

    endpoints = ListField(
        value=["http://localhost:8000/predict/teacher_model"],
        help="List of API endpoints for teacher models.",
    )
    ensemble_method = StrField(
        value="mean",
        options=["mean", "max", "first"],
        help="How to merge results from multiple teachers.",
    )
    result_json_path = StrField(
        value="result.logits",
        help="The JSON path to extract logits from the response, e.g., 'result.logits'.",
    )


@register("teacher_fetcher", "http")
class HttpTeacherFetcher(BaseTeacherFetcher):
    """Fetches teacher results via HTTP and supports ensembling multiple teachers."""

    def __init__(self, config: HttpTeacherFetcherConfig):
        super().__init__(config)
        self.config = config
        self.session = requests.Session()

    def _extract_from_json(self, data: Dict, path: str) -> Any:
        """Extracts nested value from dictionary using dot notation."""
        keys = path.split(".")
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return None
        return data

    def fetch(self, request_data: List[Dict[str, Any]]) -> List[Any]:
        """Fetches and ensembles predictions from multiple HTTP endpoints.

        Args:
            request_data: Formatted batch of input data.

        Returns:
            Ensembled logits as a list of lists/arrays.

        Raises:
            RuntimeError: If all endpoints fail or path extraction fails.
        """
        all_teacher_logits = []

        for endpoint in self.config.endpoints:
            # DLK Server is designed to accept dicts representing the raw input
            # Depending on your server API, you might need to iterate or send batch directly.
            # Here we assume the server can process a batch payload or we process one by one.
            # For simplicity in this generic fetcher, we assume sending single item loops:
            endpoint_logits = []
            try:
                for item in request_data:
                    response = self.session.post(
                        endpoint, json=item, timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    res_json = response.json()

                    logits = self._extract_from_json(
                        res_json, self.config.result_json_path
                    )
                    if logits is None:
                        raise ValueError(
                            f"Path '{self.config.result_json_path}' not found in response."
                        )
                    endpoint_logits.append(logits)

                all_teacher_logits.append(endpoint_logits)
            except Exception as e:
                logger.warning(f"Failed to fetch from teacher {endpoint}: {e}")
                continue

        if not all_teacher_logits:
            raise RuntimeError("Failed to fetch teacher labels from all endpoints.")

        # Ensemble logic
        all_teacher_logits_np = np.array(all_teacher_logits)  # [T, B, ...]
        if self.config.ensemble_method == "mean":
            ensembled = np.mean(all_teacher_logits_np, axis=0)
        elif self.config.ensemble_method == "max":
            ensembled = np.max(all_teacher_logits_np, axis=0)
        else:  # first
            ensembled = all_teacher_logits_np[0]

        return ensembled.tolist()
