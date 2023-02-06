import io
import numpy as np
import json

from typing import List
from PIL import Image
import cv2

from c_python_backend_utils import Tensor, InferenceResponse, InferenceRequest

import torchvision.transforms as transforms


class TritonPythonModel(object):
    def __init__(self):
        self.tf = None

    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')
        if len(model_config['output']) != 1:
            raise ValueError(f'Expected 1 output, got {len(model_config["output"])}')
        if 'dims' not in model_config['output'][0]:
            raise ValueError('Output dims are not defined in the model config')

        input_h = model_config['output'][0]['dims'][1]
        input_w = model_config['output'][0]['dims'][2]
        if input_h != input_w:
            raise ValueError('Output H and W must be the same')

        # TODO: transforms should be imported from classification.dataloaders
        input_size = input_h
        self.tf = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),  # to make it (input_size, input_size)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        input_name = 'input'
        output_name = 'output'

        responses = []
        for request in inference_requests:
            # This model only process one input per request. We use
            # get_input_tensor_by_name instead of checking
            # len(request.inputs()) to allow for multiple inputs but
            # only process the one we want. Same rationale for the outputs
            batch_in_tensor: Tensor = get_input_tensor_by_name(request, input_name)
            if batch_in_tensor is None:
                raise ValueError(f'Input tensor {input_name} not found '
                                 f'in request {request.request_id()}')

            if output_name not in request.requested_output_names():
                raise ValueError(f'The output with name {output_name} is '
                                 f'not in the requested outputs '
                                 f'{request.requested_output_names()}')

            batch_in = batch_in_tensor.as_numpy()  # shape (batch_size, 1)

            if batch_in.dtype.type is not np.object_:
                raise ValueError(f'Input datatype must be np.object_, '
                                 f'got {batch_in.dtype.type}')
            
            batch_out = []
            for img in batch_in:  # img is shape (1,)
                pil_img = Image.open(io.BytesIO(img.astype(bytes)))
                image = np.array(pil_img)
                if len(image.shape) == 2:  # gray image
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                np_tensor = self.tf(Image.fromarray(image, mode='RGB')).numpy()
                batch_out.append(np_tensor)

            batch_out = np.asarray(batch_out)

            # Format outputs to build an InferenceResponse
            # Assumes there is only one output
            output_tensors = [Tensor(output_name, batch_out)]

            # TODO: should set error field from InferenceResponse constructor
            # to handle errors
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses
