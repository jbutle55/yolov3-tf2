import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from tensorflow_core.python.keras.models import load_model

class_path = 'classes.names'  # Path to classes file
weights = '/checkpoints/yolov3_608.tf'  # Path to weight file
size = 608  # Resize images to size - 416 04 608
image = ''  # Path to input image
tfrecord = None  # tfrecord instead of image or None
output = ''  # Path to output image
num_classes = 3  # Number of classes in model
model_path = 'trained.h5'
loading_model = False

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

if loading_model is False:
    yolo = YoloV3(classes=num_classes)
    yolo.load_weights(weights).expect_partial()
    print('weights loaded')
else:
    yolo = load_model(model_path)

class_names = [c.strip() for c in open(class_path).readlines()]
print('classes loaded')

if tfrecord:
    dataset = load_tfrecord_dataset(tfrecord, class_path, size)
    dataset = dataset.shuffle(512)
    img_raw, _label = next(iter(dataset.take(1)))
else:
    img_raw = tf.image.decode_image(
        open(image, 'rb').read(), channels=3)

img = tf.expand_dims(img_raw, 0)
img = transform_images(img, size)

t1 = time.time()
boxes, scores, classes, nums = yolo(img)
t2 = time.time()

print('time: {}'.format(t2 - t1))
print('detections:')
for i in range(nums[0]):
    print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                       np.array(scores[0][i]),
                                       np.array(boxes[0][i])))

img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
cv2.imwrite(output, img)

print('output saved to: {}'.format(output))
