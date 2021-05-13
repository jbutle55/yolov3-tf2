import numpy as np

YOLO_MAX_BOXES = 300  # Max number of boxes per image
YOLO_IOU_THRESHOLD = 0.5
YOLO_SCORE_THRESHOLD = 0.4

IMAGE_SIZE = 608

# K-means divided by max width and height. Image size 608
# YOLO_ANCHORS = np.array([(75, 111), (99, 124), (232, 124), (302, 337), (416, 420), (490, 462),  (547, 559),
#                          (984, 1140), (1070, 1370)],
#                         np.float32) / 608

# YOLO Paper OG Anchors
# YOLO_ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#                          (59, 119), (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 608

# YOLO Shapes_BW2 Anchors - LARGE
YOLO_ANCHORS = np.array([(57, 53), (54, 63), (79, 63), (115, 163), (205, 173),
                         (225, 266), (274, 278), (358, 392), (373, 414)],
                        np.float32) / 608

# # YOLO Shapes_BW2 Anchors - SMALL
# YOLO_ANCHORS = np.array([(81, 70), (82, 75), (93, 93), (253, 244), (272, 269),
#                          (287, 306), (451, 459), (462, 466), (463, 480)],
#                         np.float32) / 608

YOLO_ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

YOLO_TINY_ANCHORS = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
YOLO_TINY_ANCHOR_MASKS = np.array([[3, 4, 5], [0, 1, 2]])


CLASS_DICT = {1: 'circle',
              2: 'triangle',
              3: 'ellipse',
              4: 'rectangle',
              5: 'square'}


# CLASS_DICT = {1: 'square',
#               2: 'circle'}

'''
CLASS_DICT = {
            1: u'car',
            2: u'bus',
            3: u'person',
            4: u'traffic light',
            5: u'motorbike',
            6: u'building',
            7: u'truck',
            8: u'train',
        }
'''


'''
CLASS_DICT = {1: u'person',
         2: u'bicycle',
         3: u'car',
         4: u'motorcycle',
         5: u'airplane',
         6: u'bus',
         7: u'train',
         8: u'truck',
         9: u'boat',
         10: u'traffic light',
         11: u'fire hydrant',
         12: u'stop sign',
         13: u'parking meter',
         14: u'bench',
         15: u'bird',
         16: u'cat',
         17: u'dog',
         18: u'horse',
         19: u'sheep',
         20: u'cow',
         21: u'elephant',
         22: u'bear',
         23: u'zebra',
         24: u'giraffe',
         25: u'backpack',
         26: u'umbrella',
         27: u'handbag',
         28: u'tie',
         29: u'suitcase',
         30: u'frisbee',
         31: u'skis',
         32: u'snowboard',
         33: u'sports ball',
         34: u'kite',
         35: u'baseball bat',
         36: u'baseball glove',
         37: u'skateboard',
         38: u'surfboard',
         39: u'tennis racket',
         40: u'bottle',
         41: u'wine glass',
         42: u'cup',
         43: u'fork',
         44: u'knife',
         45: u'spoon',
         46: u'bowl',
         47: u'banana',
         48: u'apple',
         49: u'sandwich',
         50: u'orange',
         51: u'broccoli',
         52: u'carrot',
         53: u'hot dog',
         54: u'pizza',
         55: u'donut',
         56: u'cake',
         57: u'chair',
         58: u'couch',
         59: u'potted plant',
         60: u'bed',
         61: u'dining table',
         62: u'toilet',
         63: u'tv',
         64: u'laptop',
         65: u'mouse',
         66: u'remote',
         67: u'keyboard',
         68: u'cell phone',
         69: u'microwave',
         70: u'oven',
         71: u'toaster',
         72: u'sink',
         73: u'refrigerator',
         74: u'book',
         75: u'clock',
         76: u'vase',
         77: u'scissors',
         78: u'teddy bear',
         79: u'hair drier',
         80: u'toothbrush'}
'''