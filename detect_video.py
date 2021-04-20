import time
from absl import logging
import cv2
import tensorflow as tf
from yolov3_tf2.models import  YoloV3, Darknet
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import argparse

classes_path = 'coco.names'
tiny = False
size = 1024
num_classes = 80
output = 'test-out.avi'  # Path to output video
output_format = 'XVID'
# output_format = 'MJPG'
viz_feat_map = False


def main(args):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    video = args.video
    output = args.output
    weights = args.weights

    yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes_path).readlines()]
    logging.info('classes loaded')

    if args.debug_model:
        yolo.summary()

    if args.roi_layer:
        layer_name = 'yolo_darknet'
        layer_model = Darknet(name='yolo_darknet')
        layer_model.load_weights(weights, by_name=False).expect_partial()
        yolo = layer_model

        for i in range(len(yolo.layers)):
            layer = yolo.layers[i]
            #if 'conv' not in layer.name:
            #    continue
            print(i, layer.name, layer.output_shape)

    times = []

    try:
        vid = cv2.VideoCapture(int(video))
    except:
        vid = cv2.VideoCapture(video)

    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (width, height))

    count = 0
    success = True
    while success:
        success, img = vid.read()

        print('Frame: {}'. format(count))

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, size)

        t1 = time.time()
        if args.roi_layer:
            small, medium, large = yolo.predict(img_in)
            print(small.shape)

        else:
            boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=args.thresh)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if output:
            out.write(img)
        #cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

        count = count + 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video')
    parser.add_argument('--debug_model', action='store_true')
    parser.add_argument('--roi_layer', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--roc', action='store_true')
    parser.add_argument('--weights', default='/home/justin/Models/yolov3-tf2/checkpoints/yolov3.tf')
    parser.add_argument('--thresh', default=0.5, type=int)
    arguments = parser.parse_args()
    main(arguments)
