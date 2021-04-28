import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolov3_tf2.utils import freeze_all, draw_outputs
import cv2
from collections import Counter

class Evaluator:
    """
    Evaluator class for YOLO object detection.
    Adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py
    """

    def __init__(self, iou_thresh=0.4, conf_thresh=0.5):
        """
        [summary]
        """
        self.iou_threshold = iou_thresh
        self.confidence_threshold = conf_thresh

        self.target_classes = []
        self.precision = None
        self.recall = None
        self.AP = None
        self.f1 = None
        self.unique_classes = None
        self.true_positives_list = []
        self.true_negatives = {}
        self.false_positives_list = []
        self.false_negative_list = []
        self.false_pos_count = []

    def __call__(self, outputs, ground_truths, debug_images=None):
        """
        [Summary]

        Args:
            outputs (np array): A list of all predictions for the batch.
                In the format [num_imgs x num_preds x [box coords (4) , conf, classes]]
            ground_truths (np array): A list of all ground truth annotations
                for the batch. In the format [num_imgs, 3x[num_boxes x [x1,y1,x2,y2,score,class]]
        """
        # GT actaully in shape [num imgs x 3 scales x num_boxes x (x1,y1,x2,y2,obj,class)]
        # Convert gts x,y,w,h to x1,y1,x2,y2 - passed in as form [num img's x 3 x 6]
        # ground_truths[:, :] = self.convert2xyxy(ground_truths[:, :4])

        # Boxes may already be suppresed during prediction
        # suppressed_preds = self.non_max_suppression(outputs)
        # targets?

        # Pass gts in form [num_imgs x 3 x [img no., class, x1, y1, x2, y2]]
        metrics = self.calc_batch_stats(outputs, ground_truths)
        # Returns [true_positives, pred_scores, pred_classes]

        self.true_positives, self.pred_scores, self.pred_classes = [np.concatenate(x, 0) for x in list(zip(*metrics))]
        # Flatten target classes list
        self.target_classes = [item for sub in self.target_classes for item in sub]

        self.calcMeanAveragePrecision_perClass()

        self.display_results()

        tprs, fprs = self.calc_true_negatives(outputs, ground_truths)

        self.roc_curve(tprs, fprs)

        return

    def non_max_suppression(self, predictions):
        """
        Removes any predictions with scores below the threshold limit.
        Returns predictions with shape []

        Args:
            predictions ([numpy array]): The YOLO predictions with
                shape [num_imgs x num_preds x [class, score, box coords x4]]

        Returns:
            [numpy array]: The cleaned up list of predictions following suppression
        """
        # Change center x, center y to x1,x2,y1,y2
        # The predictions are actually already in x1,y1,x2,y2 from the model
        # predictions[:, 0:3] = convert2xyxy(predictions[:, 0:3])
        output = []

        for preds in predictions:  # Loop through each images predictions
            # Filter out confidence scores below threshold
            # Maybe already done during prediction?

            preds = preds[preds[:, 4] >= self.confidence_threshold]

            if not preds.shape:  # Skip if no bounding boxes left
                continue

            # Perform non-max suppression
            keep_boxes = []
            while detects.size(0):
                # Compare first bbox to all other bboxes and strike if there
                # is an IOU larger than the threshold
                large_overlap = self.calc_bbox_iou(
                    detects[0, :4], detects[1:, :4]) > self.iou_threshold
                # Compare first bbox to all others to see if they are the same class detection
                label_match = detects[0, -1] == detects[:, -1]
                # List indicies of any bboxes with lower conf. scores, large IoUs, and matching labels
                invalid = large_overlap & label_match
                weights = detects[invalid, 4:5]
                # Merge overlapping bboxes using order of confidence using weighted sum
                detects[0::4] = (weights * detects[invalid, :4]
                                 ).sum(0) / weights.sum()
                keep_boxes += [detects[0]]
                detects = detects[~invalid]  # Delete the overlaps from detects

            if keep_boxes:
                # Stack the filtered predictions
                output[iImage] = np.stack(keep_boxes)

        return output

    def calc_bbox_iou(self, bbox_pred, bbox_target):
        """
        Calculate the IoU between two bounding boxes with coordinates x1,y1,x2,y2.

        Args:
            bbox_pred (numpy array): First bounding box coords (x1,y1,x2,y2)
            bbox_target (numpy array): Second bounding box coords [3 x (x1,y1,x2,y2)]

        Returns:
            [float]: The max calculated IoU value.
            [int]: The associated box index for the max IoU value 
        """
        iou = []

        b1_x1, b1_y1, b1_x2, b1_y2 = bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3]
        # Cycle through target boxes of all three scales
        # bbox_target = [np.squeeze(box) for box in bbox_target if len(box)]  # Remove boxes at empty scales
        # bbox_target = [ind_box for ind_box in bbox_target[0]]  # Clean up list in list

        for box in bbox_target:
            b2_x1, b2_y1, b2_x2, b2_y2 = box[0], box[1], box[2], box[3]

            # Get intersection coords
            inter_x1 = max(b1_x1, b2_x1)
            inter_y1 = max(b1_y1, b2_y1)
            inter_x2 = min(b1_x2, b2_x2)
            inter_y2 = min(b1_y2, b2_y2)

            # Calculate area of intersection rectangle
            width = inter_x2 - inter_x1
            height = inter_y2 - inter_y1

            if width < 0 or height < 0:
                # No overlap of areas
                iou.append(0.0)
                continue

            inter_area = width * height

            # Calculate union area
            b1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
            b2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))

            # Calculate IoU
            # Use 1e-16 in cases denominator == 0
            iou_calc = inter_area / (b1_area + b2_area - inter_area + 1e-16)

            iou.append(iou_calc)

        iou_max = max(iou)
        bIndex = iou.index(iou_max)

        return iou_max, bIndex

    def convert2xyxy(self, coords):
        """
        Takes coordinates in the x,y,w,h form and converts to x1,x2,y1,y2. x1,y1 corresponds to top
        left corner, x2,y2 corresponds to bottom right corner.

        Args:
            coords ([numpy array]): Box coordintes in the form centerx, centery, width, height

        Returns:
            [float]: Bounding box coordinates in form x1,y1,x2,y2
        """
        width = coords[2]
        height = coords[3]
        x1 = coords[:, 0] - width/2
        x2 = coords[0] + width/2
        y1 = coords[1] + height/2
        y2 = coords[1] - height/2

        return x1, y1, x2, y2

    def roc_curve(self, tprsm, fprs):

        return

    def calc_true_negatives(self, preds, gts):
        metrics = []

        # Count of each stat with len(num-images)
        true_pos_count = []
        false_pos_count = []
        true_neg_count = []
        false_neg_count = []
        class_tested = []

        unique_classes = None

        for iImg, outputs in enumerate(preds):
            # Complete loop for each image

            pred_boxes = outputs[:, :4]
            pred_scores = outputs[:, 4]
            pred_classes = outputs[:, 5]

            # Get annotations for the image
            annotations = gts[iImg]
            # Get the target class labels
            target_labels = []  # Should finish with shape [class, class, ...]
            for scale in annotations:
                for annot in scale:
                    target_labels.append(annot[-1])

            unique_classes = Counter(target_labels)

            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            if len(annotations):
                for single_class in unique_classes:
                    class_tested.append(single_class)

                    detected_boxes = []
                    target_boxes = []  # Should finish with shape [3 x [x1,y1,x2,y2]]
                    # Get the target bboxes
                    for scale in annotations:
                        for annot in scale:
                            if len(annot):
                                target_boxes.append(annot[:4])
                            else:
                                target_boxes.append([])

                    for iPred, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_classes)):

                        # iBox is index of gt box with max iou
                        iou, iBox = self.calc_bbox_iou(
                            pred_box, target_boxes)

                        if iou >= self.iou_threshold and iBox not in detected_boxes:
                            # Means box overlaps real object
                            if pred_label == single_class:
                                # Either True or False Positive
                                if pred_label == target_labels[iBox]:
                                    # Record the prediction as TP
                                    true_positives += 1

                                else:
                                    # False Positive
                                    false_positives += 1

                            else:
                                # Pred label != single class
                                # Either True or False Negative
                                if pred_label == target_labels[iBox]:
                                    # True Negative
                                    true_negatives += 1

                                else:
                                    # False Negative
                                    false_negatives += 1

                            detected_boxes.append(iBox)

            # Append for each class, for each image
            true_pos_count.append(true_positives)
            false_pos_count.append(false_positives)
            true_neg_count.append(true_negatives)
            false_neg_count.append(false_negatives)

        tprs = {}
        fprs = {}

        # Calculate TPR and FPR
        for ind_class in unique_classes:
            # Get indices of when ind_class in class_tested
            indices = [index for index, item in enumerate(class_tested) if item == ind_class]
            tp_value = sum([true_positives[i] for i in indices])
            fp_value = sum([false_positives[i] for i in indices])
            tn_value = sum([true_negatives[i] for i in indices])
            fn_value = sum([false_negatives[i] for i in indices])

            if tp_value == 0:
                tpr = 0
            else:
                tpr = tp_value / (tp_value + fn_value)

            if fp_value == 0:
                fpr = 0
            else:
                fpr = fp_value / (fp_value + tn_value)

            tprs[ind_class] = tpr
            fprs[ind_class] = fpr

            print(f'Class: {ind_class}, TPR: {tpr}, FPR: {fpr}')

        return [tprs, fprs]

    def calc_batch_stats(self, preds, gts):
        """[summary]

        Args:
            preds ([type]): [description]
            gts ([type]): [description]

        Returns:
            [type]: [description]
        """

        metrics = []

        for iImg, outputs in enumerate(preds):
            # Complete loop for each image
            if len(outputs) == 0:
                # No predictions to evaluate
                continue

            #outputs = preds[iSample]
            pred_boxes = outputs[:, :4]
            pred_scores = outputs[:, 4]
            pred_classes = outputs[:, 5]

            per_class_preds = {}  # Count number of correct preds for each class
            true_negs = {}

            # Create list of zeros same size as predictions for TPs
            true_positives = np.zeros(pred_boxes.shape[0])

            # Get annotations for the image
            annotations = gts[iImg]
            # Get the target class labels
            target_labels = []  # Should finish with shape [class, class, ...]
            for scale in annotations:
                print(f'scale: {scale}')
                for annot in scale:
                    print(f'annot: {annot}')
                    target_labels.append(annot[-1])

            self.target_classes.append(target_labels)

            if len(annotations):
                detected_boxes = []
                detected_classes = []
                target_boxes = []  # Should finish with shape [3 x [x1,y1,x2,y2]]
                # TODO Shape of target_boxes might be wrong
                # Get the target bboxes
                for scale in annotations:
                    for annot in scale:
                        if len(annot):
                            target_boxes.append(annot[:4])
                        else:
                            target_boxes.append([])

                false_pos_count = 0
                for iPred, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_classes)):
                    # If target found, break
                    if len(detected_boxes) == len(annotations):
                        break

                    # Ignore class if not the target class
                    if pred_label not in target_labels:
                        # Means false positive (before suppression)
                        false_pos_count = false_pos_count + 1
                        continue

                    # iBox is index of gt box with max iou
                    iou, iBox = self.calc_bbox_iou(
                        pred_box, target_boxes)
                    
                    # Ensure only GTs of class in question are used
                    match = (target_labels == pred_label) & (iou >= self.iou_threshold)
                    
                    if iou >= self.iou_threshold and iBox not in detected_boxes:
                        if pred_label == target_labels[iBox]:
                            # Record the prediction as TP
                            true_positives[iPred] = 1
                            detected_boxes.append(iPred)  # Append index of gt boxes not coords
                            detected_classes.append(pred_classes[iPred])

            metrics.append([true_positives, pred_scores, pred_classes])

        return metrics

    def calcMeanAveragePrecision_perClass(self):
        tps = self.true_positives
        conf = self.pred_scores
        pred_classes = self.pred_classes
        target_classes = self.target_classes

        # Sort by descending objectness
        i = np.argsort(-conf)
        tp, conf, pred_classes = tps[i], conf[i], pred_classes[i]

        # Find unique classes
        unique_classes = np.unique(target_classes)

        ap, p, r = [], [], []
        self.true_positives_list, self.false_positives_list = [], []
        self.false_negative_list = []
        for c in tqdm.tqdm(unique_classes, desc='Computing AP'):
            i = pred_classes == c  # Get positive matches of classes with pred_classes
            num_gt = (target_classes == c).sum()  # Get number of groundtruth for class
            num_pred = i.sum()  # Get number of positive predictions

            if num_pred == 0 and num_gt == 0:
                continue
            elif num_pred == 0 and num_gt != 0:
                ap.append(0)
                p.append(0)
                r.append(0)

                self.true_positives_list.append(0)
                self.false_positives_list.append(0)

            else:
                fp_count = (1 - tp[i]).cumsum()  # Get num. of false positives
                tp_count = (tp[i]).cumsum()  # Get num. of true positives

                # Recall
                # Calculate recall values
                recall_curve = tp_count / (num_gt + 1e-16)
                r.append(recall_curve[-1])

                # Precision
                precision_curve = tp_count / (tp_count + fp_count)  # Calc precision values
                p.append(precision_curve[-1])

                # AP from recall-precision curve
                ap.append(self.calc_ap(recall_curve, precision_curve))

                self.true_positives_list.append(tp_count[-1])
                self.false_positives_list.append(fp_count[-1])

        # Compute F1 score
        p, r, ap = np.array(p), np.array(r), np.array(ap)
        f1 = 2 * p * 4 / (p + r + 1e-16)

        self.precision = p
        self.recall = r
        self.AP = ap
        self.f1 = f1
        self.unique_classes = unique_classes.astype('int32')

        return p, r, ap, f1, unique_classes.astype('int32')

    def calc_ap(self, recall, precision):
        """
        Compute the average precision for a given recall and precision curves.

        Args:
            recall (list): The recall curve
            precision (list): The precision curve

        Returns:
            [type]: The computed average precision
        """
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Plot precision-recall graph
        # self.plot_precision_recall(mpre, mrec)

        # Compute precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

        # self.plot_precision_recall(mpre, mrec)

        # Find points along x-axis where PR changes direction
        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

        return ap

    def plot_precision_recall(self, precision, recall):
        plt.plot(recall, precision, 'r+')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
        return

    def display_results(self):
        text_file = open('Eval_Results.txt', 'w')

        # Find better way to do this so not requiring manual changes
        #class_dict = {0: 'car', 1: 'bus', 2: 'truck', 3: 'person',
        #              4: 'bicycle', 5: 'motorbike', 6: 'train',
        #              7: 'building', 8: 'traffic light'}

        class_dict = {1: u'person',
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

        print('Average Precisions: ')
        for i, c in enumerate(self.unique_classes):
            print('Class {} - AP: {}'.format(class_dict[c+1], self.AP[i+1]))
            text_file.write('Class {} - AP: {} \n'.format(class_dict[c], self.AP[i]))
            text_file.write('True Positives - {}\n False Positives - {}\n\n'.format(self.true_positives_list,
                                                                                    self.false_positives_list))

        print('mAP: {}'.format(self.AP.mean()))
        text_file.write('mAP: {}'.format(self.AP.mean()))

        text_file.close()

        return
