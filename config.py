import numpy as np

YOLO_MAX_BOXES = 300  # Max number of boxes per image
YOLO_IOU_THRESHOLD = 0.2
YOLO_SCORE_THRESHOLD = 0.1

IMAGE_SIZE = 608

# K-means divided by max width and height. Image size 608
YOLO_ANCHORS = np.array([(75, 111), (99, 124), (232, 124), (302, 337), (416, 420), (490, 462),  (547, 559),
                         (984, 1140), (1070, 1370)],
                        np.float32) / 608

# YOLO Paper OG Anchors
YOLO_ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 608

YOLO_ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

YOLO_TINY_ANCHORS = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
YOLO_TINY_ANCHOR_MASKS = np.array([[3, 4, 5], [0, 1, 2]])