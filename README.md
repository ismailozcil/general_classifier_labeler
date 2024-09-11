# Multi Label or Classification Estimation using Pre-Trained Networks

This code performs multi-label or classification estimation on images using a pre-trained convolutional neural network (CNN). It identifies objects in an image using YOLO trained on MSCOCO and predicts their possible labels.
## Usage

1.  **Install required libraries:**
    Install "geomloss" and "ultralytics" libraries. Both are available for pip installation.
    $pip install geomloss
    $pip install ultralytics
3.  **Define the model and dataset parameters:**
    *   `model_name`: The name of the pre-trained model to use (e.g., 'resnet18', 'regnet_y').
    *   `dataset_name`: The dataset used for training (e.g., 'CIFAR100').
    *   `img_size`: The size of the input image (e.g., 224).
5.  **Create an instance of the `ImageLabeler` class:**
    $python labeler = ImageLabeler(model_name=model_name, dataset_name=dataset_name, img_size=img_size)
6.  **Load the pre-trained models:**
    $python labeler.load_models()
7.  **Get an image from the webcam or use a downloaded image to be labeled:**
    To get an image from the webcam: $python labeler.getWebcamPhoto(imagename='photo.jpg')
8.   **Perform multi-label or classification estimation:**
    $python labeler.image_estimate(imagename='photo.jpg')
9.   **The code will display the image with bounding boxes around the detected objects and their predicted label/s.**

## Note

*   Optional/ This labeler requires a GPU for optimal performance.
*   The code uses YOLOv8 for object segmentation.
*   The code includes functions for calculating curvature and optimizing thresholds.
*   The `get_opt_result` function calculates the optimal labels based on scores and thresholds.
*   The `image_estimate` function performs the main estimation process.
*   The code includes visualization functions for plotting ROC curves and displaying results.
   

