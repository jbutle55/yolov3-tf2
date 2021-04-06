import os
import sys
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yolov3_tf2 import models
import tensorflow as tf

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from yolov3_tf2.models import YoloV3, YoloV3Tiny, YoloLoss, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
from yolov3_tf2.utils import freeze_all, draw_outputs
from yolov3_tf2.dataset import transform_images, transform_targets, load_tfrecord_dataset
from eval_utils import Evaluator



def main():

    train_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/Shapes/tfrecord_single/coco_train.record-00000-of-00001'
    valid_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/Shapes/tfrecord_single/coco_val.record-00000-of-00001'
    weights_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/yolov3-tf2/checkpoints/yolov3.tf'
    # Path to text? file containing all classes, 1 per line
    classes = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/yolov3-tf2/shapes/shapes.names'
    # Usually fit
    # mode = 'fit'  # Can be 'fit', 'eager_fit', 'eager_tf', 'valid'
    mode = 'fit'
    '''
    'fit: model.fit, '
    'eager_fit: model.fit(run_eagerly=True), '
    'eager_tf: custom GradientTape'
    '''

    # Usually darknet
    transfer = 'none'
    '''
    'none: Training from scratch, '
                      'darknet: Transfer darknet, '
                      'no_output: Transfer all but output, '
                      'frozen: Transfer and freeze all, '
                      'fine_tune: Transfer all and freeze darknet only'),
                      'pre': Use a pre-trained model for validation
    '''
    image_size = 416
    num_epochs = 1
    batch_size = 8
    learning_rate = 1e-3
    num_classes = 4
    # num class for `weights` file if different, useful in transfer learning with different number of classes
    weight_num_classes = 80
    iou_threshold = 0.5

    # saved_weights_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/yolov3-tf2/weights/'
    saved_weights_path = '/home/justin/ml_models/yolov3-tf2/weights/shapes_{}.tf'.format(num_epochs)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # Training dataset
    #dataset_train = tf.data.TFRecordDataset(train_path)
    #dataset_val = tf.data.TFRecordDataset(valid_path)

    dataset_train = load_tfrecord_dataset(train_path, classes, image_size)
    dataset_train = dataset_train.shuffle(buffer_size=512)
    dataset_train = dataset_train.batch(batch_size)
    #dataset_train = dataset_train.map(lambda x, y: (
    #    transform_images(x, image_size),
    #    transform_targets(y, anchors, anchor_masks, image_size)))
    #dataset_train = dataset_train.prefetch(
    #    buffer_size=tf.data.experimental.AUTOTUNE)

    dataset_val = load_tfrecord_dataset(valid_path, classes, image_size)
    dataset_val = dataset_val.shuffle(buffer_size=512)
    dataset_val = dataset_val.batch(batch_size)
    #dataset_val = dataset_val.map(lambda x, y: (
    #    transform_images(x, image_size),
    #    transform_targets(y, anchors, anchor_masks, image_size)))

    # Create model in training mode
    yolo = models.YoloV3(image_size, training=True, classes=num_classes)

    model_pretrained = YoloV3(image_size,
                              training=True,
                              classes=weight_num_classes or num_classes)
    model_pretrained.load_weights(weights_path)

    # Which weights to start with?
    print('Loading Weights...')
    #yolo.load_weights(weights_path)

    yolo.get_layer('yolo_darknet').set_weights(
        model_pretrained.get_layer('yolo_darknet').get_weights())
    freeze_all(yolo.get_layer('yolo_darknet'))

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in anchor_masks]  # Passing loss as a list might sometimes fail? dict might be better?

    yolo.compile(optimizer=optimizer, loss=loss,
                              run_eagerly=(mode == 'eager_fit'))
    callbacks = [ReduceLROnPlateau(verbose=1),
                 EarlyStopping(patience=3, verbose=1),
                 ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
                 TensorBoard(log_dir='logs')]

    history = yolo.fit(dataset_train,
                        epochs=num_epochs,
                        callbacks=callbacks,
                        validation_data=dataset_val)
    yolo.save_weights(saved_weights_path)


    # Detect/ROC
    model = YoloV3(image_size, training=False, classes=num_classes)
    model.load_weights(saved_weights_path).expect_partial()

    batch_size = 1

    val_dataset = load_tfrecord_dataset(valid_path, classes, image_size)
    val_dataset = val_dataset.batch(batch_size)

    val_dataset = val_dataset.map(lambda x, y: (
        transform_images(x, image_size),
        transform_targets(y, anchors, anchor_masks, image_size)))

    images = []
    for img, labs in val_dataset:
        img = np.squeeze(img)
        images.append(img)

    predictions = []

    evaluator = Evaluator(iou_thresh=iou_threshold)

    # labels - (N, grid, grid, anchors, [x, y, w, h, obj, class])
    boxes, scores, classes, num_detections = model.predict(val_dataset)
    # boxes -> (num_imgs, num_detections (200), box coords (4))
    # scores -> (num_imgs, num_detections)
    # classes -> (num_imgs, num_detections)
    # num_detections -> num_imgs

    # Aim for labels shape (per batch): [num_imgs, 3x[num_boxes x [x1,y1,x2,y2,score,class]]
    # full_labels = [label for _, label in val_dataset]

    # Shape : [Num images, 3 scales, grid, grid, anchor, 6 ]

    filtered_labels = []

    for _, label in val_dataset:
        img_labels = []
        # Label has shape [3 scales x[1, grid, grid, 3, 6]]
        for scale in label:
            # Shape [1, grid, grid, 3, 6]
            scale = np.asarray(scale)
            grid = scale.shape[1]

            scale2 = np.reshape(scale, (3, grid * grid, 6))
            # Shape: [3, grix*grid, 6]

            for anchor in scale2:
                filtered_anchors = []
                for box in anchor:
                    if box[4] > 0:
                        filtered_anchors.append(np.asarray(box))
            img_labels.append(filtered_anchors)

        img_labels = np.asarray(img_labels)
        filtered_labels.append(img_labels)

    print(len(filtered_labels))
    print(len(filtered_labels[0]))
    print(len(filtered_labels[0][2]))

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

    # Predictions shape: [num_imgs x num_preds x[box coords(4), conf, classes]]
    # Box coords should be in format x1 y1 x2 y2

    # Labels shape: [num_imgs, 3x[num_boxes x [x1,y1,x2,y2,score,class]]
    evaluator(predictions, filtered_labels, images)  # Check gts box coords


    '''
    # Detect
    confidence_thresholds = [0.5]
    tp_rates = []
    fp_rates = []

    # Compute ROCs for above range of thresholds
    # Compute 1 for each class vs. the other classes
    for index, conf in confidence_thresholds:

        tp_at_conf = 0
        fp_at_conf = 0

        
        image_ids = np.random.choice(dataset_val.image_ids, 10)
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]

            # Need TPR and FPR rates for each class versus the other classes

            # Compute TPR (recall)
            _, _, tpr, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r["rois"], r["class_ids"], r["scores"], r['masks'])

            fpr = utils.compute_fpr()

            tp_at_conf += tpr
            fp_at_conf += fpr

        tp_rates.append(tp_at_conf)
        fp_rates.append(fp_at_conf)

    # Plot roc curves
    utils.compute_roc_curve()


    image_ids = np.random.choice(dataset_val.image_ids, 10)
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    '''

if __name__ == '__main__':
    main()
