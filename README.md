# Image Processing with YOLO Models

This project demonstrates the integration of YOLO object detection and segmentation models to process images, draw bounding boxes around detected objects, apply segmentation masks, and combine these visualizations in various ways. It utilizes the Ultralytics YOLO implementation and OpenCV for image manipulation.

## Project Structure

- `main.py`: The main script that orchestrates image loading, prediction, and visualization.
- `const.py`: Contains constants used across the project, including model paths and color definitions for labels.
- `utils.py`: Provides utility classes and functions for image processing tasks, such as matching images with their labels, drawing boxes and masks, and combining images.

### Key Components

- **ImageLabelMatcher**: Matches images with their corresponding labels from specified directories.
- **BoxDrawer**: Contains methods for drawing detection boxes and segmentation masks on images, combining images with captions, and superposing images.
- **Color Definitions**: Unique colors are defined for different labels to visually distinguish between various detected objects and segmented areas.

## Setup

1. Ensure you have Python 3.6+ installed.
2. Install required packages:

```bash
pip install -r requirements.txt
```
for cuda 12:
```bash
pip install -r requirements_cuda.txt
```
3. Place your images and labels in the specified directories as defined in `const.py`. The directories should be structured with separate subfolders for images and labels.

## Running the Script

Execute the `main.py` script:

```bash
python main.py
```

This will process images from the specified dataset path, apply object detection and segmentation, and display the results. For each image, detection boxes and segmentation masks are drawn, and images are combined or superposed according to the script logic.

### Functionality

- **Detection and Segmentation**: The script uses pre-trained YOLO models for object detection and segmentation, drawing bounding boxes and masks on the images.
- **Image Combination**: Detected and segmented images are combined side by side with captions indicating the type of processing applied.
- **Image Superposition**: Additionally, detected and segmented images are superposed to visualize the overlay of detection and segmentation results.
- **Interactive Viewing**: Images are displayed sequentially, proceeding to the next pair upon a key press, with all windows closing after the final image.

## Customization

To adapt the project for different datasets or models:
- Update paths in `const.py` to point to your models and datasets.
- Modify color schemes in `const.py` to suit different labels or preferences.
- Adjust image processing functions in `utils.py` as needed for specific requirements.

This project offers a flexible framework for experimenting with object detection and segmentation models on custom datasets.
