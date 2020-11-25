import sys

sys.path.append("stubs/")
sys.path.append("/Users/ratzzz/Documents/tensorrt-inference-server/client_server/inf_ser/utils/")

from tritongrpcclient import grpc_service_pb2, grpc_service_pb2_grpc
import tritongrpcclient.model_config_pb2 as model_config
import inference_pb2
import inference_pb2_grpc
import utils
import logging as log
from PIL import Image
import io
import grpc
import numpy as np
import tensorflow as tf


class Segmentation(inference_pb2_grpc.InferenceAPIServicer):
    """ This class returns the segmentation result of the input image. """
    def __init__(self):
        """ Set Parameters corresponding the gRPC end point of the tensorrt inference server. """
        self._IMAGE_WIDTH = 0
        self._IMAGE_HEIGHT = 0
        # Within the Docker container => trt-server:8001
        # Without Docker container => localhost:8001
        self._URL = "localhost:8001"
        self._MODEL_NAME = "unet"
        self._BATCH_SIZE = 1
        # -1 points the model to the latest version in the model repository.
        self._MODEL_VERSION = ""

    def prepare_request(self, img):
        """
        Prepare gRPC request for model inference with the tensorrt inference server.

        Arguments:
        img(PIL.Image): Input image.

        Returns:
        request(gRPC Request object): Request object with all information about the request.
        """
        request = grpc_service_pb2.ModelInferRequest()
        request.model_name = self._MODEL_NAME
        request.model_version = self._MODEL_VERSION
        output = grpc_service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name = "conv2d_27/truediv"
        request.outputs.extend([output])
        #print('Hello')

        input = grpc_service_pb2.ModelInferRequest().InferInputTensor()
        input.name = "input_4"
        input.datatype = "FP32"
        input.shape.extend([self._BATCH_SIZE, 256, 512, 3])

        image_data = []
        image_data.append(self.unet_preprocess(img))
        input_bytes = image_data[0].tobytes()

        input_contents = grpc_service_pb2.InferTensorContents()
        input_contents.raw_contents = input_bytes
        input.contents.CopyFrom(input_contents)
        request.inputs.extend([input])

        return request

    def unet_preprocess(self, img):
        """
        Preprocess the input image.

        Arguments:
        img(PIL.Image): Input image.

        Returns:
        img(numpy array): Preprocessed 3d numpy array of the input image.
        """
        img_rgb = img.convert("RGB")

        self._IMAGE_WIDTH, self._IMAGE_HEIGHT = img.size
        img_arr = np.array(img.getdata()).reshape(
            (self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 3)
        )
        # img_arr = np.expand_dims(img_arr, axis=0)
        # UNET model normalizes the input between -1 and +1.
        img_arr = img_arr / 127.5 - 1

        # UNET model expects input data type as float32.
        img_arr = img_arr.astype(np.float32)

        return img_arr

    def result_map_to_img(self, pred):

        color_map = {
            '0': [0, 0, 0],
            '1': [196, 8, 206],
            '2': [27, 22, 186],
            '3': [242, 4, 4],
            '4': [66,244,235],
            '5': [231,207,127],
            '6': [5, 79, 3]
        }

        res_map = np.zeros((256, 512, 3), dtype=np.uint8)

        argmax_idx = np.argmax(pred, axis=2)

        for i in range(0, 256):
          for j in range(0, 512):
            res_map[i, j] = color_map[str(argmax_idx[i, j])]

        return res_map

    def segmentation_request(self, img):
        """
        Sends request to the running model at the tensorrt inference server and returns the response.

        Arguments:
        img(PIL.Image) : Input image.

        Returns:
        index(int) : Index of the predicted class.
        """
        with grpc.insecure_channel(self._URL) as channel:
            grpc_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel)
            request = self.prepare_request(img)
            response = grpc_stub.ModelInfer(request)
            result = response.outputs[0].contents.raw_contents

            res_array = np.frombuffer(result, dtype=np.float32).reshape((256, 512, 7))
            res_map = self.result_map_to_img(res_array)

            return res_map.tobytes()

    def GetSegmentation(self, request, context):
        """
        This function takes request from the request dispatcher and sends back the response to it.

        Arguments:
        request : gRPC request object.
        context : gRPC context object.

        Returns:
        resp : Segmentation map in bytes.
        """
        seg_map = inference_pb2.ImageResponse()
        img = utils.Utils.bytes_to_image(self, request.image)
        response = self.segmentation_request(img)
        seg_map.res_map = response

        return seg_map
