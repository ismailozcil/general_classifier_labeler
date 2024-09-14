#!/usr/bin/env python3
import numpy as np
from PIL import Image 
import cv2

import torch 
from ultralytics import YOLO 


class ObjectDetectorYOLO:
    """
    Detects objects in an image using a YOLOv5 model.

    Initializes an object detector with a specified YOLOv5 model,
    loads the pre-trained model and detects objects in an input image.

    Args:
        model_name: The name of the YOLOv5 model to use (default: 'yolov5s').

    Attributes:
        model_name: The name of the YOLOv5 model.
        model: The loaded YOLOv5 model.
    """
    def __init__(self, model_name='yolov5s'):
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        self.model_name = model_name
        # Load the pretrained YOLOv5 model.
        self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)

    def detect_objects(self, image_path):
        """
        Detects objects in an image.

        Passes the input image through the YOLOv5 model and returns the detected objects.

        Args:
            image_path: A PIL Image or a path to the image.

        Returns:
            A list of detected objects, each represented as a dictionary with a bounding box and label information.
        """

        if not isinstance(image_path, (str, Image.Image)):
            raise TypeError("Input image must be a PIL Image or a path to the image.")
        # Detect objects in the image.
        results = self.model(image_path)
        # Return the cropped objects with bounding boxes.
        return results.crop(save=False)

class ObjectSegmenterYOLO:
    """
    Detects and segments objects in an image using a YOLOv8 segmentation model.

    Initializes an object segmenter with a specified YOLOv8 model,
    loads the model and defines the class names and IDs.

    Args:
        model_name: The name of the YOLOv8 segmentation model to use (default: 'yolov8m-seg.pt').

    Attributes:
        model_name: The name of the YOLOv8 segmentation model.
        model: The loaded YOLOv8 segmentation model.
        yolo_classes: A list of class names used by the YOLOv8 model.
        classes_ids: A list of class IDs corresponding to the class names.
    """
    def __init__(self, model_name='yolov8m-seg.pt'):
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")

        self.model_name = model_name
        # Load the YOLOv8 segmentation model.
        self.model = YOLO(self.model_name)
        # Get the class names used by the YOLOv8 model.
        self.yolo_classes = list(self.model.names.values())
        # Get the class IDs corresponding to the class names.
        self.classes_ids = [self.yolo_classes.index(clas) for clas in self.yolo_classes]

    def detect_objects(self, image_path):
        """
        Detects and segments objects in an image.

        Passes the input image through the YOLOv8 segmentation model and returns the results.

        Args:
            image_path: The path to the image.

        Returns:
            The results of object detection and segmentation, including masks and bounding boxes.
        """
        if not isinstance(image_path, (str, Image.Image)):
            raise TypeError("Input image must be a PIL Image or a path to the image.")
        # Detect and segment objects in the image.
        results = self.model(image_path)
        # Return the results.
        return results[0]


