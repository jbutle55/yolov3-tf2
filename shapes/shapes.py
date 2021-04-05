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


    # Full labels shape -> [num_batches, grid scale, imgs]
    # Full labels shape -> [num_batches, [grid, grid, anchors, [x,y,w,h,obj,class]]]
    full_labels = np.asarray([label for _, label in val_dataset])
    full_labels = [label for _, label in val_dataset]  # Shape [num_imgs x 3 [1, 13, 13, 3, 6]

    test = np.empty(len(full_labels))

    # Aim Labels shape: [num_imgs, 3x[num_boxes x [x1,y1,x2,y2,score,class]]

    for img in full_labels:
        for scale in img:
            scale = np.empty(1)
            less_dim = scale[0]
            for grid1 in less_dim:
                for grid2 in grid1:
                    for anchor in grid2:
                        x1 = anchor[0]
                        y1 = anchor[1]
                        w = anchor[2]
                        h = anchor[3]
                        obj = anchor[4]
                        cls = anchor[5]

                        x2 = x1 + w
                        y2 = y1 + h

                        if obj != 0:
                            vect = [x1, y1, x2, y2, obj, cls]
                            np.append(scale, vect)







    # Shape -> [num_batches, num_imgs_in_batch, 3]
    # Shape -> [num_batches, num_imgs, 3x[grid,grid,anchors,[x,y,w,h,score,class]]]
    full_labels_trans = full_labels.transpose(0, 2, 1)

    full_labels_flat = []

    for batch in full_labels_trans:
        for img in batch:
            row = []
            for scale in img:
                row.append(scale)
            full_labels_flat.append(row)

    # Shape -> [num_imgs x 3]
    full_labels_flat = np.asarray(full_labels_flat)

    # Remove any labels consisting of all 0's
    filt_labels = []
    # for img in range(len(full_labels_flat)):
    for img in full_labels_flat:
        test = []
        # for scale in full_labels_flat[img]:
        for scale in img:
            lab_list = []
            for g1 in scale:
                for g2 in g1:
                    for anchor in g2:
                        if anchor[0] > 0:
                            temp = [anchor[0] * image_size,
                                    anchor[1] * image_size,
                                    anchor[2] * image_size,
                                    anchor[3] * image_size,
                                    anchor[4],
                                    anchor[5]]
                            temp = [float(x) for x in temp]
                            lab_list.append(np.asarray(temp))
            test.append(np.asarray(lab_list))
        filt_labels.append(np.asarray(test))
    filt_labels = np.asarray(filt_labels)  # Numpy array of shape [num_imgs, 3x[num_boxesx[x1,y1,x2,y2,score,class]]]
    # filt_labels = filt_labels[:, :4] * image_size

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
    evaluator(predictions, filt_labels, images)  # Check gts box coords


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
