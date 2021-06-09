import time
import cv2
import argparse
import numpy as np
from collections import Counter

import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import YoloV3, YoloLoss, yolo_anchors, yolo_anchor_masks
from yolov3_tf2.utils import freeze_all, draw_outputs
from eval_utils import Evaluator
import config as cfg
import tensorflow as tf
import matplotlib.pyplot as plt


# Flatten all labels and strip labels containing all 0's
def flatten_labels(label, image_size):
    filt_labels = []

    # Flatten all labels and remove 0s
    for size in label:
        for grid1 in size:
            for grid2 in grid1:
                for anchor in grid2:
                    for a in anchor:
                        if a[4] > 0:
                            temp = [a[0] * image_size,
                                    a[1] * image_size,
                                    a[2] * image_size,
                                    a[3] * image_size,
                                    a[4],
                                    a[5]]
                            temp = [float(x) for x in temp]
                            filt_labels.append(np.asarray(temp))
    filt_labels = np.asarray(filt_labels)

    return filt_labels


def main(args):

    image_size = 608  # 416
    batch_size = 1
    num_classes = 80
    # num class for `weights` file if different, useful in transfer learning with different number of classes
    valid_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/aerial-cars-private/aerial_yolo/train/train.tfrecord'
    weights_path = 'checkpoints/yolov3_608.tf'
    # Path to text? file containing all classes, 1 per line
    classes = 'coco.names'
    iou = 0.5

    anchors = cfg.YOLO_ANCHORS
    anchor_masks = cfg.YOLO_ANCHOR_MASKS

    val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                classes,
                                                image_size)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, image_size),
        dataset.transform_targets(y, anchors, anchor_masks, image_size)))

    model = YoloV3(image_size,
                   training=False,
                   classes=num_classes)
    model.load_weights(weights_path)

    confidence_thresholds = np.linspace(0.1, 1, 15)
    confidence_thresholds = [0.1, 0.3, 0.5, 0.7]
    # confidence_thresholds = [0.1]

    all_tp_rates = []
    all_fp_rates = []

    evaluator = Evaluator(iou_thresh=iou, test_data=val_dataset)

    class_dict = cfg.CLASS_DICT
    class_names = list(class_dict.values())

    tp_rates = {}
    fp_rates = {}

    # Compute ROCs for above range of thresholds
    # Compute one for each class vs. the other classes
    for index, conf in enumerate(confidence_thresholds):

        tp_of_img = []
        fp_of_img = []
        all_classes = []

        # tp_rates = {}
        # fp_rates = {}

        boxes, scores, classes, num_detections = model.predict(val_dataset)

        visual_preds = False
        if visual_preds:

            index = 0
            for img_raw, _label in val_dataset.take(5):
                print(f'Index {index}')

                # img = tf.expand_dims(img_raw, 0)
                img = dataset.transform_images(img_raw, image_size)
                img = img * 255

                boxes, scores, classes, nums = model(img)

                output = 'test_images/test_{}.jpg'.format(index)
                # output = '/Users/justinbutler/Desktop/test/test_images/test_{}.jpg'.format(index)

                # print('detections:')
                # for i in range(nums[index]):
                #     print('\t{}, {}, {}'.format(class_names[int(classes[index][i])],
                #                               np.array(scores[index][i]),
                #                               np.array(boxes[index][i])))
                #     if i > 10:
                #         continue

                img = cv2.cvtColor(img_raw[0].numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=0)
                img = img * 255
                cv2.imwrite(output, img)

                index = index + 1

        visual_gts = False
        if visual_gts:
            index = 0
            for img_raw, _label in val_dataset.take(5):
                print(f'Index {index}')
                # img = tf.expand_dims(img_raw, 0)
                img = dataset.transform_images(img_raw, image_size)

                output = 'test_images/test_labels_{}.jpg'.format(index)
                # output = '/Users/justinbutler/Desktop/test/test_images/test_labels_{}.jpg'.format(index)

                filt_labels = flatten_labels(_label, image_size)

                boxes = tf.expand_dims(filt_labels[:, 0:4], 0)
                scores = tf.expand_dims(filt_labels[:, 4], 0)
                classes = tf.expand_dims(filt_labels[:, 5], 0)
                nums = tf.expand_dims(filt_labels.shape[0], 0)

                img = cv2.cvtColor(img_raw[0].numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=0)
                img = img * 255

                cv2.imwrite(output, img)

                index = index + 1

        filtered_labels = []
        for _, label in val_dataset:
            filt_labels = flatten_labels(label, image_size)
            filtered_labels.append(filt_labels)

        predictions = []

        # i is the num_images index
        # predictions = [np.hstack([boxes[i][x], scores[i][x], classes[i][x]]) for i in range(len(num_detections)) for x in range(len(scores[i])) if scores[i][x] > 0]
        for img in range(len(num_detections)):
            row = []
            for sc in range(len(scores[img])):
                if scores[img][sc] > 0:
                    row.append(np.hstack([boxes[img][sc] * image_size, scores[img][sc], classes[img][sc]]))
            predictions.append(np.asarray(row))

        predictions = np.asarray(predictions)  # numpy array of shape [num_imgs x num_preds x 6]

        if len(predictions) == 0:  # No predictions made
            print('No predictions made - exiting.')
            exit()

        # predictions[:, :, 0:4] = predictions[:, :, 0:4] * image_size
        # Predictions format - [num_imgs x num_preds x [box coords x4, score, classes]]
        # Box coords should be in format x1 y1 x2 y2

        evaluator(predictions, filtered_labels, roc=True)  # Check gts box coords

        tprs = evaluator.true_pos_rate
        fprs = evaluator.false_pos_rate

        tp_rates[conf] = tprs  # Dict of Dicts
        fp_rates[conf] = fprs

    for k in evaluator.classes_counter:
        tps = []
        fps = []
        for conf in confidence_thresholds:
            tps.append(tp_rates[conf][k])
            fps.append(fp_rates[conf][k])

        print(f'TPS: {tps}, FPS: {fps}')

        plt.plot(tps, fps, 'r+')
        plt.title(f'ROC - {class_dict[k+1]}')
        plt.xlabel('True Pos. Rate')
        plt.ylabel('False Pos Rate')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    arguments = parser.parse_args()
    main(arguments)