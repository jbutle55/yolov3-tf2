import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
# from tensorflow_core.python.keras.models import load_model

class_path = 'shapes.names'  # Path to classes file
weights = 'shapes_bw_200ep.tf'  # Path to weight file
image_size = 608  # Resize images to size - 416 04 608
image = ''  # Path to input image
tfrecord = '/home/justin/Data/Shapes_BlackWhite/tf_records/coco_val.tfrecord-00000-of-00001'  # tfrecord instead of image or None
output = '/home/justin/test/test_images'  # Path to output image
num_classes = 5  # Number of classes in model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

yolo = YoloV3(image_size, training=False, classes=num_classes)
yolo.load_weights(weights).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(class_path).readlines()]
print('classes loaded')

if tfrecord:
    dataset = load_tfrecord_dataset(tfrecord, class_path, image_size)
    dataset = dataset.shuffle(512)
    img_raw, _label = next(iter(dataset.take(1)))
else:
    img_raw = tf.image.decode_image(
        open(image, 'rb').read(), channels=3)

index = 0
for img_raw, _label in dataset.take(5):
    # img = tf.expand_dims(img_raw, 0)
    img = transform_images(img_raw, image_size)
    img = img * 255

    boxes, scores, classes, nums = yolo(img)

    output = 'test_images/test_{}.jpg'.format(index)
    print('output saved to: {}'.format(output))

    img = cv2.cvtColor(img_raw[0].numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=0)
    img = img * 255
    cv2.imwrite(output, img)

    index = index + 1

