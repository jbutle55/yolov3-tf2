import time
import cv2
import argparse
import numpy as np

import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import YoloV3, YoloLoss, yolo_anchors, yolo_anchor_masks
from yolov3_tf2.utils import freeze_all, draw_outputs
from eval_utils import Evaluator


def main(args):

    image_size = 416  # 416
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 1e-3
    num_classes = args.num_classes
    # num class for `weights` file if different, useful in transfer learning with different number of classes
    weight_num_classes = args.num_weight_class
    valid_path = args.valid_dataset
    weights_path = args.weights
    # Path to text? file containing all classes, 1 per line
    classes = args.classes

    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                classes,
                                                image_size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, image_size),
        dataset.transform_targets(y, anchors, anchor_masks, image_size)))

    model = YoloV3(image_size, training=True, classes=num_classes)
    # Darknet transfer is a special case that works
    # with incompatible number of classes
    # reset top layers
    model_pretrained = YoloV3(image_size,
                              training=True,
                              classes=weight_num_classes or num_classes)
    model_pretrained.load_weights(weights_path)

    if transfer == 'darknet':
        model.get_layer('yolo_darknet').set_weights(
            model_pretrained.get_layer('yolo_darknet').get_weights())
        freeze_all(model.get_layer('yolo_darknet'))

    predictions = []

    evaluator = Evaluator(iou_thresh=args.iou)

    # labels - (N, grid, grid, anchors, [x, y, w, h, obj, class])
    boxes, scores, classes, num_detections = model.predict(val_dataset)
    # boxes -> (num_imgs, num_detections, box coords)

    # Full labels shape -> [num_batches, grid scale, imgs]
    # Full labels shape -> [num_batches, [grid, grid, anchors, [x,y,w,h,obj,class]]]
    full_labels = np.asarray([label for _, label in val_dataset])

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

    # predictions[:, :, 0:4] = predictions[:, :, 0:4] * image_size
    # Predictions format - [num_imgs x num_preds x [box coords x4, score, classes]]
    # Box coords should be in format x1 y1 x2 y2

    evaluator(predictions, filt_labels, images)  # Check gts box coords


    confidence_thresholds = np.linspace(0.1, 1, 15)
    confidence_thresholds = [0.5]
    all_tp_rates = []
    all_fp_rates = []

    # Compute ROCs for above range of thresholds
    # Compute one for each class vs. the other classes
    for index, conf in enumerate(confidence_thresholds):
        tp_of_img = []
        fp_of_img = []
        all_classes = []

        tp_rates = {}
        fp_rates = {}

        boxes, scores, classes, num_detections = model.predict(val_dataset)

        # Full labels shape -> [num_batches, grid scale, imgs]
        # Full labels shape -> [num_batches, [grid, grid, anchors, [x,y,w,h,obj,class]]]
        full_labels = np.asarray([label for _, label in val_dataset])

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
        filt_labels = np.asarray(
            filt_labels)  # Numpy array of shape [num_imgs, 3x[num_boxesx[x1,y1,x2,y2,score,class]]]
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

        # predictions[:, :, 0:4] = predictions[:, :, 0:4] * image_size
        # Predictions format - [num_imgs x num_preds x [box coords x4, score, classes]]
        # Box coords should be in format x1 y1 x2 y2

        evaluator(predictions, filt_labels, images)  # Check gts box coords


        classes = list(set(r['class_ids']))  # All unique class ids
        for c in classes:
            if c not in all_classes:
                all_classes.append(c)
        complete_classes = dataset_val.class_ids[1:]
        # Need TPR and FPR rates for each class versus the other classes
        # Recall == TPR
        tpr = utils.compute_ap_indiv_class(gt_bbox, gt_class_id, gt_mask,
                                           r["rois"], r["class_ids"], r["scores"], r['masks'], complete_classes)
        total_fpr = utils.compute_fpr_indiv_class(gt_bbox, gt_class_id, gt_mask,
                                                  r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                  complete_classes)
        # print(f'For Image: TPR: {tpr} -- FPR: {total_fpr}')
        tp_of_img.append(tpr)
        fp_of_img.append(total_fpr)

        all_classes = dataset_val.class_ids[1:]

        # Need to get average TPR and FPR for number of images used
        for c in all_classes:
            tp_s = 0
            for item in tp_of_img:
                if c in item.keys():
                    tp_s += item[c]
                else:
                    tp_s += 0

            tp_rates[c] = tp_s / len(image_ids)
            # tp_rates[c] = tp_s

        # print(tp_rates)

        for c in all_classes:
            fp_s = 0
            for item in fp_of_img:
                if c in item.keys():
                    fp_s += item[c]
                else:
                    fp_s += 0
            fp_rates[c] = fp_s / len(image_ids)
            # fp_rates[c] = fp_s

        all_fp_rates.append(fp_rates)
        all_tp_rates.append(tp_rates)

    print(f'TP Rates: {all_tp_rates}')
    print(f'FP Rates: {all_fp_rates}')

    # Plot roc curves
    # utils.compute_roc_curve(all_tp_rates, all_fp_rates, save_fig=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    arguments = parser.parse_args()
    main(arguments)