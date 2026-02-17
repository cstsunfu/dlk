# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import hjson
import pandas as pd
import torch
from fastapi import FastAPI, Request
from intc import (
    MISSING,
    Base,
    BoolField,
    DictField,
    IntField,
    NestField,
    StrField,
    cregister,
    dataclass,
)
from ray import serve
from ray.serve.handle import DeploymentHandle

from dlk.predict import Predict
from dlk.preprocess import PreProcessor
from dlk.utils.register import register

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

logger = logging.getLogger(__name__)
app = FastAPI()


@dataclass
class ModelServeConfig(Base):
    """Configuration for a single model's deployment pipeline."""

    process_config = StrField(
        value=MISSING, help="Path to the preprocessor config file."
    )
    fit_config = StrField(
        value=MISSING, help="Path to the fit config file for training."
    )
    checkpoint = StrField(value=MISSING, help="Path to the trained model checkpoint.")
    use_onnx = BoolField(
        value=False, help="Whether to use the ONNX model instead of PyTorch."
    )
    onnx_path = StrField(
        value="", help="Path to the ONNX model (required if use_onnx=True)."
    )

    class ResourceConfig:
        """Configuration for scaling and resource allocation of a specific deployment."""

        num_replicas = IntField(
            value=1,
            help="Number of replicas for this deployment.",
        )
        num_cpus = IntField(
            value=1,
            help="Number of CPUs allocated per replica.",
        )
        num_gpus = IntField(
            value=0,
            help="Number of GPUs allocated per replica (0 for CPU only).",
        )
        max_concurrent_queries = IntField(
            value=100,
            help="Maximum number of concurrent queries per replica.",
        )
        max_batch_size = IntField(
            value=8,
            help="Maximum batch size for inference (only effective on Inference deployment).",
        )
        batch_wait_timeout_s = FloatField(
            value=0.1,
            help="Maximum time to wait to form a batch (only effective on Inference deployment).",
        )

    preprocessor_resources = NestField(
        value=ResourceConfig, help="Resources for the preprocessor stage."
    )
    inference_resources = NestField(
        value=ResourceConfig, help="Resources for the inference stage."
    )
    postprocessor_resources = NestField(
        value=ResourceConfig, help="Resources for the postprocessor stage."
    )


@cregister("ray_serve")
class RayServeGlobalConfig(Base):
    """Global configuration for the Ray Serve application."""

    models = DictField(
        value={},
        help="Dictionary of model names to their ModelServeConfig.",
    )


@serve.deployment
class DLKPreprocessor:
    """Ray Serve Deployment for data preprocessing.

    Handles tokenization, image transformations, and feature gathering.
    Runs primarily on CPU.
    """

    def __init__(self, config_dict: Dict):
        """Initializes the Preprocessor.

        Args:
            config_dict: Dictionary representation of ModelServeConfig.
        """
        self.config: ModelServeConfig = ModelServeConfig._from_dict(config_dict)
        self.processor = PreProcessor(
            hjson.load(open(self.config.process_config, "r", encoding="utf-8")),
            stage="online",
        )

        # We use Predict class just to easily initialize the Datamodule for formatting
        self.predict_helper = Predict(self.config.fit_config, self.config.checkpoint)
        self.datamodule, _ = self.predict_helper.get_datamodule(
            self.predict_helper.dlk_config, {}, world_size=1
        )

    async def __call__(
        self, input_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Processes raw input data.

        Args:
            input_data: Raw JSON dictionary from the HTTP request.

        Returns:
            A tuple containing:
            - The uncollated processed data item (suitable for passing to Inference).
            - The original input data (needed for postprocessing).
        """
        # Convert single dict to DataFrame for compatibility with dlk.preprocess
        input_df = pd.DataFrame([input_data])
        processed_data = self.processor.fit(input_df)

        # Extract the single processed item to pass to the dynamic batcher
        columns = list(processed_data.keys())
        key_type_pairs = self.datamodule.dataset_creator.real_key_type_pairs(
            self.datamodule.dataset_config.key_type_pairs, columns
        )

        # Create a lightweight dataset wrapper for the single item
        dataset = self.datamodule.dataset_creator(
            self.datamodule.dataset_config,
            processed_data,
            self.datamodule.rt_config,
            key_type_pairs,
        )
        uncollated_item = dataset[0]

        return uncollated_item, input_data


@serve.deployment
class DLKInference:
    """Ray Serve Deployment for Neural Network Inference.

    Utilizes `@serve.batch` to dynamically group concurrent individual requests
    into a single batch to maximize GPU/CPU throughput.
    """

    def __init__(self, config_dict: Dict):
        """Initializes the Inference model (PyTorch or ONNX).

        Args:
            config_dict: Dictionary representation of ModelServeConfig.
        """
        self.config: ModelServeConfig = ModelServeConfig._from_dict(config_dict)
        self.predict_helper = Predict(self.config.fit_config, self.config.checkpoint)

        self.datamodule, _ = self.predict_helper.get_datamodule(
            self.predict_helper.dlk_config, {}, world_size=1
        )

        self.use_onnx = self.config.use_onnx
        if self.use_onnx:
            if onnxruntime is None:
                raise ImportError("onnxruntime is required for ONNX inference.")
            self.ort_session = onnxruntime.InferenceSession(
                self.config.onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.onnx_input_names = [inp.name for inp in self.ort_session.get_inputs()]
            self.onnx_output_names = [
                out.name for out in self.ort_session.get_outputs()
            ]
        else:
            self.model = self.predict_helper.imodel
            self.model.eval()
            if (
                torch.cuda.is_available()
                and self.config.inference_resources.num_gpus > 0
            ):
                self.model = self.model.cuda()

    @serve.batch
    async def __call__(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Performs batched inference.

        Args:
            batch_items: A list of uncollated items gathered by Ray Serve.

        Returns:
            A list of dictionary outputs, corresponding 1:1 with the input list.
        """
        # 1. Collate the list of items into a tensor batch
        collated_batch = self.datamodule.collate_fn(batch_items, stage="online")

        # 2. Move to device if using PyTorch
        if (
            not self.use_onnx
            and torch.cuda.is_available()
            and self.config.inference_resources.num_gpus > 0
        ):
            for k, v in collated_batch.items():
                if isinstance(v, torch.Tensor):
                    collated_batch[k] = v.cuda()

        # 3. Perform Inference
        if self.use_onnx:
            ort_inputs = {
                k: v.cpu().numpy()
                for k, v in collated_batch.items()
                if k in self.onnx_input_names
            }
            ort_outs = self.ort_session.run(None, ort_inputs)

            # Reconstruct dictionary output
            outputs = {
                name: torch.tensor(out)
                for name, out in zip(self.onnx_output_names, ort_outs)
            }
        else:
            with torch.no_grad():
                outputs = self.model.predict_step(collated_batch, 0)

        # 4. Split the batched output back into a list of individual dicts
        return self._split_batch_dict(outputs, len(batch_items))

    def _split_batch_dict(
        self, batch_output: Dict[str, torch.Tensor], batch_size: int
    ) -> List[Dict[str, Any]]:
        """Splits a batched dictionary of tensors into a list of individual dictionaries."""
        results = []
        for i in range(batch_size):
            item = {}
            for k, v in batch_output.items():
                item[k] = v[i] if isinstance(v, torch.Tensor) and v.ndim > 0 else v
            results.append(item)
        return results


@serve.deployment
class DLKPostprocessor:
    """Ray Serve Deployment for data postprocessing.

    Converts model logits/ids into human-readable JSON formats and calculates metrics if needed.
    """

    def __init__(self, config_dict: Dict):
        """Initializes the Postprocessor.

        Args:
            config_dict: Dictionary representation of ModelServeConfig.
        """
        self.config = ModelServeConfig._from_dict(config_dict)
        self.predict_helper = Predict(self.config.fit_config, self.config.checkpoint)
        self.postprocessor = self.predict_helper.imodel.postprocessor

    async def __call__(
        self, model_output: Dict[str, Any], origin_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Postprocesses the output."""
        batched_output = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
            for k, v in model_output.items()
        }

        origin_data_list = [origin_data]

        result_list = self.postprocessor.wrap_predict_one_batch(
            stage="online",
            batch_output=batched_output,
            origin_data=origin_data_list,
            rt_config={},
        )
        return result_list[0]


@serve.deployment
@serve.ingress(app)
class APIGateway:
    """FastAPI Ingress Deployment.

    Routes HTTP requests to the respective model pipelines, allowing for both
    single-model queries and multi-model ensembling.
    """

    def __init__(self, model_pipelines: Dict[str, Dict[str, DeploymentHandle]]):
        """Initializes the Gateway.

        Args:
            model_pipelines: A nested dictionary mapping model_name -> {'pre': handle, 'infer': handle, 'post': handle}.
        """
        self.model_pipelines = model_pipelines

    async def _execute_pipeline(
        self, model_name: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes the full Pre -> Infer -> Post pipeline for a specific model."""
        if model_name not in self.model_pipelines:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.model_pipelines.keys())}"
            )

        handles = self.model_pipelines[model_name]

        # 1. Preprocess
        uncollated_item, original_data = await handles["pre"].remote(data)

        # 2. Infer (will be dynamically batched by Ray Serve)
        model_output = await handles["infer"].remote(uncollated_item)

        # 3. Postprocess
        final_result = await handles["post"].remote(model_output, original_data)

        return final_result

    @app.post("/predict/{model_name}")
    async def predict(self, model_name: str, request: Request):
        """Endpoint for querying a single model."""
        data = await request.json()
        try:
            result = await self._execute_pipeline(model_name, data)
            return {"status": "success", "model": model_name, "result": result}
        except Exception as e:
            logger.exception("Error during prediction")
            return {"status": "error", "message": str(e)}

    @app.post("/ensemble")
    async def ensemble(self, request: Request):
        """Endpoint for querying multiple models concurrently (Ensembling).

        Request Body Example:
        {
            "models": ["bert_model", "roberta_model"],
            "data": {"sentence": "This is a test."}
        }
        """
        req_json = await request.json()
        target_models = req_json.get("models", [])
        data = req_json.get("data", {})

        if not target_models:
            return {
                "status": "error",
                "message": "No target models specified for ensemble.",
            }

        # Execute pipelines concurrently
        tasks = [
            self._execute_pipeline(model_name, data) for model_name in target_models
        ]
        try:
            results = await asyncio.gather(*tasks)
            ensemble_result = {model: res for model, res in zip(target_models, results)}
            return {"status": "success", "ensemble_results": ensemble_result}
        except Exception as e:
            logger.exception("Error during ensemble prediction")
            return {"status": "error", "message": str(e)}


def build_app(config_file: str) -> serve.Application:
    """Builds and wires the Ray Serve Application Graph.

    Args:
        config_file: Path to the RayServeGlobalConfig JSON/HJSON file.

    Returns:
        The bound Ray Serve Application ready to be deployed.
    """
    config_dict = hjson.load(open(config_file, "r", encoding="utf-8"))
    global_config = RayServeGlobalConfig._from_dict(config_dict["@ray_serve"])

    model_pipelines = {}

    for model_name, model_config_dict in global_config.models.items():
        m_config: ModelServeConfig = ModelServeConfig._from_dict(model_config_dict)

        # Instantiate Preprocessor
        pre_opts = m_config.preprocessor_resources
        pre_deployment = DLKPreprocessor.options(
            num_replicas=pre_opts.num_replicas,
            ray_actor_options={
                "num_cpus": pre_opts.num_cpus,
                "num_gpus": pre_opts.num_gpus,
            },
            max_concurrent_queries=pre_opts.max_concurrent_queries,
            name=f"{model_name}_pre",
        ).bind(model_config_dict)

        # Instantiate Inference (with batching parameters)
        infer_opts = m_config.inference_resources
        # We dynamically inject batching properties into the decorator using `.options`
        # Ray 2.x allows modifying batching params in options if defined in the class.
        DLKInference_configured = DLKInference.options(
            num_replicas=infer_opts.num_replicas,
            ray_actor_options={
                "num_cpus": infer_opts.num_cpus,
                "num_gpus": infer_opts.num_gpus,
            },
            max_concurrent_queries=infer_opts.max_concurrent_queries,
            name=f"{model_name}_infer",
        )
        # Update batch size dynamically
        DLKInference_configured._batch_size = infer_opts.max_batch_size
        DLKInference_configured._batch_wait_timeout_s = infer_opts.batch_wait_timeout_s
        infer_deployment = DLKInference_configured.bind(model_config_dict)

        # Instantiate Postprocessor
        post_opts = m_config.postprocessor_resources
        post_deployment = DLKPostprocessor.options(
            num_replicas=post_opts.num_replicas,
            ray_actor_options={
                "num_cpus": post_opts.num_cpus,
                "num_gpus": post_opts.num_gpus,
            },
            max_concurrent_queries=post_opts.max_concurrent_queries,
            name=f"{model_name}_post",
        ).bind(model_config_dict)

        model_pipelines[model_name] = {
            "pre": pre_deployment,
            "infer": infer_deployment,
            "post": post_deployment,
        }

    # Bind the Gateway with the constructed pipelines
    gateway = APIGateway.bind(model_pipelines)
    return gateway
