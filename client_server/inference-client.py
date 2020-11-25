""" Inference client (python)"""
from __future__ import print_function
import logging

import grpc
import sys
import argparse
sys.path.append("./inf_ser/stubs/")

import inference_pb2
import inference_pb2_grpc
import numpy as np
from PIL import Image

def get_image_in_bytes(img_path):

    """
    This function converts the image to bytes.

    Parameters:
    img_path(str) : path to the image.

    Returns:
    bytes : image in bytes.
    """

    with open(img_path, "rb") as f:
        img = f.read()

    return img


def run(img_path):

    """
    This function performs model inference and prints the result.

    Parameters:
    img_path(str) : path to the image.

    """

    with grpc.insecure_channel("localhost:50053") as channel:
        stub = inference_pb2_grpc.InferenceAPIStub(channel)

        for _ in range(0, 1):
            # response = stub.GetClassification(
            #     inference_pb2.ImageRequest(image=get_image_in_bytes(img_path))
            # )
            response = stub.GetSegmentation(
                inference_pb2.ImageRequest(image=get_image_in_bytes(img_path))
            )

        final_map = np.frombuffer(response.res_map, dtype=np.uint8).reshape((256, 512, 3))
        final_map_img = Image.fromarray(final_map)
        final_map_img.save("segmentation_map.jpeg")

def get_args():

    """
    Gets the path to the image from the user.

    Returns:
    str : image path entered by the user
    """
    parser = argparse.ArgumentParser(description='Inference client.')
    parser.add_argument("--img_path", required=True, type=str, help="Path to the image for inference.")
    args = parser.parse_args()

    return args.img_path

if __name__ == "__main__":
    logging.basicConfig()
    img_path = get_args()
    run(img_path)
