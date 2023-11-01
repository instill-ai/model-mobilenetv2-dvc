import io
from typing import List
import ray
import torch
import requests

import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
from instill.configuration import CORE_RAY_ADDRESS
from instill.helpers.ray_helper import (
    DataType,
    serialize_byte_tensor,
    deserialize_bytes_tensor,
    deploy_decorator,
    undeploy_decorator,
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

ray.init(address=CORE_RAY_ADDRESS)
# this import must come after `ray.init()`
from ray import serve


@serve.deployment()
class MobileNet:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        self.categories = self._image_labels()
        self.model = ort.InferenceSession(model_path)
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


@deploy_decorator
def deploy_model(
    num_cpus: str,
    num_replicas: str,
    application_name: str,
    model_path: str,
    model_name: str,
    route_prefix: str,
):
    c_app = MobileNet.options(
        name=application_name,
        ray_actor_options={
            "num_cpus": num_cpus,
        },
        num_replicas=num_replicas,
    ).bind(model_path)

    serve.run(c_app, name=model_name, route_prefix=route_prefix)


@undeploy_decorator
def undeploy_model(model_name: str):
    serve.delete(model_name)


if __name__ == "__main__":
    args = entry()

    if args.func == "deploy":
        deploy_model(
            num_cpus=args.cpus, num_replicas=args.replicas, model_path=args.model
        )
    elif args.func == "undeploy":
        undeploy_model(model_path=args.model)
