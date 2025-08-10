# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import time

import hjson
import onnx
import onnxruntime
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
    Parser,
    StrField,
    SubModule,
    cregister,
    dataclass,
)
from ray import serve
from starlette.requests import Request

from dlk.export import ExportConfig
from dlk.preprocess import PreProcessor
from dlk.utils.import_module import import_module_dir
from dlk.utils.register import register


@cregister("ray_serve")
class RayServeConfig(Base):
    """the base loss config"""

    process_config = StrField(
        value=MISSING, help="Path to the preprocessor config file."
    )
    fit_config = StrField(
        value=MISSING, help="Path to the fit config file for training."
    )
    export_config = StrField(
        value=MISSING,
        help="Path to save the exported model after training.",
    )

    class RayServeInstanceConfig:
        num_replicas = IntField(
            value=1,
            additions=["auto"],
            help="Number of replicas for the preprocessor deployment. Defaults to 1. If 'auto', the number of replicas will be determined by the autoscaling configuration.",
        )
        ray_actor_options = DictField(
            value=1,
            additions=[1],
            help="Options to pass to the Ray Actor decorator, such as resource requirements. Valid options are: `accelerator_type`, `memory`, `num_cpus`, `num_gpus`, `resources`, and `runtime_env`.",
        )
        max_replicas_per_node = IntField(
            value=None,
            additions=[None],
            help="The max number of replicas of this deployment that can run on a single node. Valid values are None (default, no limit) or an integer in the range of [1, 100]. This cannot be set together with placement_group_bundles.",
        )
        max_ongoing_requests = IntField(
            value=5,
            help="Maximum number of requests that can be sent to a preprocessor replica without receiving a response. Defaults to 5.",
        )
        max_queued_requests = IntField(
            value=-1,
            help="Maximum number of requests to this preprocessor deployment that will be queued at each caller (proxy or DeploymentHandle). Once this limit is reached, subsequent requests will raise a BackPressureError (for handles) or return an HTTP 503 status code (for HTTP requests). Defaults to -1 (no limit).",
        )

    process_instance_config = NestField(
        value=RayServeInstanceConfig,
        help="Configuration for the preprocessor deployment.",
    )
    fit_instance_config = NestField(
        value=RayServeInstanceConfig,
        help="Configuration for the fit deployment.",
    )
    postprocess_instance_config = NestField(
        value=RayServeInstanceConfig,
        help="Configuration for the postprocessor deployment.",
    )


class RayServe(object):
    """docstring for RayServe"""

    def __init__(self, config: RayServeConfig):
        super(RayServe, self).__init__()
        self.config: RayServeConfig = config

        preprocess_instance = PreProcessor(
            hjson.load(open(self.config.process_config, "r", encoding="utf-8")),
            stage="online",
            update_config={},
        )

        export_config: ExportConfig = ExportConfig._from_dict(
            hjson.load(open(self.config.export_config, "r", encoding="utf-8"))
        )
        ort_session = onnxruntime.InferenceSession(export_config.output_path)

        fit_config = Parser(
            hjson.load(open(self.config.fit_config, "r", encoding="utf-8")),
        ).parser_init()[0]["@fit"]["@imodel"]["@postprocessor"]


@serve.deployment(num_replicas=2)
class Preprocessor:
    def __init__(self, inference_handle: "serve.DeploymentHandle"):
        self.inference_handle = inference_handle
        print("Preprocessor initialized.")

    async def handle(self, request: Request) -> str:
        data = await request.json()
        print(f"Preprocessor received data: {data}")

        # 1. 预处理 (CPU密集型)
        heavy_cpu_work(0.1)  # 模拟0.1秒的预处理
        preprocessed_data = f"preprocessed({data})"
        print(f"Data preprocessed: {preprocessed_data}")

        # 2. 异步调用推理部署，并立刻开始处理下一个请求
        # 调用返回一个 ObjectRef，我们 await 它来获取最终结果
        print("Forwarding to inference model...")
        model_output_ref = self.inference_handle.run.remote(preprocessed_data)

        # 在这里，当前 worker 可以处理新请求了！
        # await 将控制权交还，直到结果准备好
        final_result = await model_output_ref

        print(f"Final result received: {final_result}")
        return final_result


@serve.deployment(ray_actor_options={"num_gpus": 1})
class InferenceModel:
    def __init__(self, postprocessor_handle: "serve.DeploymentHandle"):
        self.postprocessor_handle = postprocessor_handle
        print("InferenceModel initialized.")
        # self.model = load_model().to("cuda")

    # 使用 @serve.batch 实现自动批处理，最大化GPU利用率
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def run(self, inputs: list[str]) -> list[str]:
        print(f"InferenceModel received a batch of size: {len(inputs)}")

        # 3. 模型推理 (GPU密集型)
        # batch_tensors = self.model(inputs)
        heavy_gpu_work(0.5)  # 模拟0.5秒的推理
        print(f"Inference complete for batch: {inputs}")

        # 4. 异步调用后处理
        # 这里用 for 循环为批次中的每个项目发起一个后处理请求
        post_refs = [
            self.postprocessor_handle.run.remote(f"inferred({item})") for item in inputs
        ]

        # 并行等待所有后处理完成
        return await asyncio.gather(*post_refs)


# --- 部署 3: 后处理 ---
@serve.deployment(num_replicas=2)
class Postprocessor:
    def __init__(self):
        print("Postprocessor initialized.")

    async def run(self, data: str) -> str:
        print(f"Postprocessor received data: {data}")

        # 5. 后处理 (CPU密集型)
        heavy_cpu_work(0.05)  # 模拟0.05秒的后处理
        result = f"postprocessed({data})"
        print(f"Data postprocessed: {result}")
        return result


def main(config):
    postprocessor = Postprocessor.bind()
    inference_model = InferenceModel.bind(postprocessor)
    # 2. 将句柄注入上游部署，构建成一个图
    app = Preprocessor.bind(inference_model)
