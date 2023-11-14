import io
from typing import List
import ray
import torch
import requests

import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
from instill.helpers.const import DataType
from instill.helpers.ray_io import serialize_byte_tensor, deserialize_bytes_tensor
from instill.helpers.ray_config import (
    InstillRayModelConfig,
    get_compose_ray_address,
    entry,
)

from ray_pb2 import (
    ModelReadyRequest,
    ModelReadyResponse,
    ModelMetadataRequest,
    ModelMetadataResponse,
    ModelInferRequest,
    ModelInferResponse,
    InferTensor,
)

ray.init(address=get_compose_ray_address(10001))
# this import must come after `ray.init()`
from ray import serve


@serve.deployment()
class MobileNet:
    def __init__(self, model_path: str):
        self.categories = self._image_labels()
        self.model = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        self.tf = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _image_labels(self) -> List[str]:
        categories = []
        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        labels = requests.get(url, timeout=10).text
        for label in labels.split("\n"):
            categories.append(label.strip())
        return categories

    def process_model_outputs(self, output: np.array):
        probabilities = torch.nn.functional.softmax(torch.from_numpy(output), dim=0)
        prob, catid = torch.topk(probabilities, 1)

        return catid, prob

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="onnx",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="input",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="output",
                    shape=[1000],
                ),
            ],
        )
        return resp

    def ModelReady(self, req: ModelReadyRequest) -> ModelReadyResponse:
        resp = ModelReadyResponse(ready=True)
        return resp

    async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
        b_tensors = request.raw_input_contents[0]

        input_tensors = deserialize_bytes_tensor(b_tensors)

        batch_out = []
        for enc in input_tensors:
            img = Image.open(io.BytesIO(enc.astype(bytes)))
            image = np.array(img)
            np_tensor = self.tf(Image.fromarray(image, mode="RGB")).numpy()
            batch_out.append(np_tensor)

        batch_out = np.asarray(batch_out)
        out = self.model.run(None, {"input": batch_out})
        # shape=(1, batch_size, 1000)

        # tensor([[207], [294]]), tensor([[0.7107], [0.7309]])
        cat, score = self.process_model_outputs(out[0])
        s_out = [
            bytes(f"{score[i][0]}:{self.categories[cat[i]]}", "utf-8")
            for i in range(cat.size(0))
        ]

        out = serialize_byte_tensor(np.asarray(s_out))
        out = np.expand_dims(out, axis=0)

        return ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
            outputs=[
                InferTensor(
                    name="output",
                    shape=[len(batch_out), 1000],
                ),
            ],
            raw_output_contents=out,
        )


def deploy_model(model_config: InstillRayModelConfig):
    c_app = MobileNet.options(
        name=model_config.application_name,
        ray_actor_options=model_config.ray_actor_options,
        max_concurrent_queries=model_config.max_concurrent_queries,
        autoscaling_config=model_config.ray_autoscaling_options,
    ).bind(model_config.model_path)

    serve.run(
        c_app, name=model_config.model_name, route_prefix=model_config.route_prefix
    )


def undeploy_model(model_name: str):
    serve.delete(model_name)


if __name__ == "__main__":
    func, model_config = entry("model.onnx")

    model_config.ray_actor_options["num_cpus"] = 2
    model_config.ray_actor_options["num_gpus"] = 0.5

    if func == "deploy":
        deploy_model(model_config=model_config)
    elif func == "undeploy":
        undeploy_model(model_name=model_config.model_name)
