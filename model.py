import io
from typing import List
import torch
import requests

import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
from instill.helpers.const import DataType
from instill.helpers.ray_io import serialize_byte_tensor, deserialize_bytes_tensor
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


@instill_deployment
class MobileNet:
    def __init__(self):
        self.categories = self._image_labels()
        self.model = ort.InferenceSession(
            "model.onnx", providers=["CUDAExecutionProvider"]
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

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="input",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="output",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1000],
                ),
            ],
        )
        return resp

    async def __call__(self, req):
        b_tensors = req.raw_input_contents[0]

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
        score, cat = torch.topk(torch.from_numpy(out[0]), 1)
        s_out = [
            bytes(f"{score[i][0]}:{self.categories[cat[i]]}", "utf-8")
            for i in range(cat.size(0))
        ]

        out = serialize_byte_tensor(np.asarray(s_out))
        out = np.expand_dims(out, axis=0)

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="output",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[len(batch_out), 1000],
                )
            ],
            raw_outputs=out,
        )


entrypoint = InstillDeployable(MobileNet).get_deployment_handle()
