import os
import random
import cv2
from ultralytics import YOLO
from utils import ImageLabelMatcher, BoxDrawer
from const import *

def load_matched_files(images_path=PATH_BOX_DATASET):  
    try:return ImageLabelMatcher(images_path).get_matched_files()
    except FileNotFoundError as e:
        print(e)
        exit()
        
def draw_black_boxes_from_predictions(image, predictions_list):
    for prediction in predictions_list:
        box = prediction['box']
        text = f"{prediction['label']} ({prediction['confidence']:.2f})"
        color = DETECT_LABELS_COLORS.get(prediction['class'], (0, 0, 0))
        image = BoxDrawer.draw_box_on_image(image, box, text=text, fill=True, box_color=color)
    return image

def draw_segments_from_predictions(image, predictions_list):
    for prediction in predictions_list:
        text = f"{prediction['label']} ({prediction['confidence']:.2f})"
        color = SEGMENT_LABELS_COLORS.get(prediction['class'], (0, 0, 0))
        image = BoxDrawer.draw_mask_on_image(image, prediction['mask'], text=text, mask_color=color)
    return image           

def format_detect_predictions(predictions):
            boxes, classes, confidences = (predictions.boxes.xywhn.cpu().numpy(),predictions.boxes.cls.cpu().numpy(),predictions.boxes.conf.cpu().numpy())
            return  [{'box': box,'class': cls,
                'confidence': conf,'label': detect_model_labels.get(cls)} 
                for box, cls, conf in zip(boxes, classes, confidences)]
            
def format_segment_predictions(predictions):
    masks, classes, confidences = (predictions.masks.xy,predictions.boxes.cls.cpu().numpy(),predictions.boxes.conf.cpu().numpy())
    return [{ 'mask': masks[i] ,'class': classes[i],
            'confidence': confidences[i],'label': segment_model_labels.get(classes[i])}
            for i in range(len(classes))]

if __name__ == "__main__":
    images_path = os.path.join(PATH_SEG_DATASET, 'train')
    images_pairs = load_matched_files(images_path)
    detect_model = YOLO(DETECT_10K_YOLOV8M_100EPOCH)
    segment_model = YOLO(SEGMENT_YOLOV8M_SEG_5PEOCH)
    detect_model_labels = detect_model.names
    segment_model_labels = segment_model.names
    random.shuffle(images_pairs)

for image_path, _ in images_pairs:  
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
    if original_image is None: continue
    
    box_prediction = detect_model.predict(original_image, imgsz=640, conf=0.1, device=DEVICE)
    if box_prediction: detect_predictions_list = format_detect_predictions(box_prediction[0])
    box_image = original_image.copy()
    box_image = draw_black_boxes_from_predictions(box_image, detect_predictions_list)
    
    seg_prediction= segment_model.predict(original_image, imgsz=640, conf=0.1, device=DEVICE)
    if seg_prediction[0]: segment_predictions_list = format_segment_predictions(seg_prediction[0])
    seg_img = original_image.copy()
    seg_img = draw_segments_from_predictions(seg_img, segment_predictions_list)
    
    combined_img = BoxDrawer.combine_images_with_captions(box_image, "Object Detection", seg_img, "Segmentation")
    mixed_img = BoxDrawer.superpose_images(box_image, seg_img)
    
    # Display the combined and superposed images in separate windows
    cv2.imshow("Combined", combined_img)
    cv2.imshow("Superposed", mixed_img)
    
    # Wait indefinitely for a key press before proceeding to the next set of images
    cv2.waitKey(0)

# After the loop, destroy all windows
cv2.destroyAllWindows()
