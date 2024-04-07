DETECT_10K_YOLOV8N_100EPOCH     = r"D:\tonobilat\propre_detection\tonobilat\models\best_boxDetect_10k_yolov8n_100epoch.pt"
DETECT_10K_YOLOV8M_100EPOCH     = r"D:\tonobilat\propre_detection\tonobilat\models\best_boxDetect_10k_yolov8m_100epoch.pt"
DETECT_10K_YOLOV8S_100EPOCH     = r"D:\tonobilat\propre_detection\tonobilat\models\best_boxDetect_10k_yolov8s_100epoch.pt"
SEGMENT_YOLOV8N_SEG_100PEOCH    = r"D:\tonobilat\propre_detection\tonobilat\models\best_segdata_yolov8n_seg_100epoch.pt"
SEGMENT_YOLOV8M_SEG_5PEOCH      = r"D:\tonobilat\propre_detection\tonobilat\models\best_segdata_yolov8m_seg_5epoch.pt"

PATH_BOX_DATASET                = r"D:\tonobilat"
PATH_SEG_DATASET                = r"D:\tonobilat\Lost And Found Seg H-NH.v2-v2-with-augmentation.yolov8"

DEVICE = "cpu" # "cpu" or "0" if using GPU available
IMG_SIZE = 640 # pixels
WAIT_TIME = 2500  # ms

DETECT_LABELS_COLORS = {
    0: (64, 128, 128),  # Sombre Teal for 'bus'
    1: (128, 64, 0),    # Sombre Orange for 'car'
    2: (128, 128, 64),  # Olive Green for 'rider'
    3: (64, 0, 128),    # Sombre Purple for 'traffic light'
    4: (64, 128, 64),   # Dark Sea Green for 'traffic sign'
    5: (128, 64, 128),  # Plum for 'train'
    6: (0, 105, 128)    # Dark Slate Blue for 'truck'
}

SEGMENT_LABELS_COLORS = {
    0: (0, 255, 255),   # Cyan for 'Hazard'
    1: (255, 105, 180), # Hot Pink for 'Non-Hazard'
    2: (255, 165, 0),   # Orange for 'border'
    3: (173, 255, 47),  # Green Yellow for 'free space'
    4: (255, 20, 147)   # Deep Pink for 'undefined'
}

