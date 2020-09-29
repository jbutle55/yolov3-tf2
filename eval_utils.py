import numpy as np
import tqdm
import matplotlib.pyplot as plt


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
        self.false_positives_list = []
        self.false_negative_list = []

    def __call__(self, outputs, ground_truths):
        """
        [Summary]

        Args:
            outputs (np array): A list of all predictions for the batch.
                In the format [num_imgs x [box coords, conf, classes]]
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
            inter_area = abs((inter_x2 - inter_x1) * (inter_y2 - inter_y1))

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

            # Create list of zeros same size as predictions for TPs
            true_positives = np.zeros(pred_boxes.shape[0])

            # Get annotations for the image
            annotations = gts[iImg]
            # Get the target class labels
            target_labels = []  # Should finish with shape [class, class, ...]
            for scale in annotations:
                for annot in scale:
                    target_labels.append(annot[-1])

            self.target_classes.append(target_labels)

            if len(annotations):
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
                    # If target found, break
                    if len(detected_boxes) == len(annotations):
                        break

                    # Ignore class if not the target class
                    if pred_label not in target_labels:
                        # Means false positive
                        continue

                    iou, iBox = self.calc_bbox_iou(
                        pred_box, target_boxes)
                    
                    # Ensure only GTs of class in question are used
                    # match = (target_labels == pred_label) & (iou >= self.iou_threshold)
                    
                    if iou >= self.iou_threshold and iBox not in detected_boxes:
                        # Record the prediction as TP
                        true_positives[iPred] = 1
                        detected_boxes.append(iBox)
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
            elif num_pred == 0 or num_gt == 0:
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
        class_dict = {0: 'car', 1: 'bus', 2: 'truck', 3: 'person',
                      4: 'bicycle', 5: 'motorbike', 6: 'train',
                      7: 'building', 8: 'traffic light'}

        print('Average Precisions: ')
        for i, c in enumerate(self.unique_classes):
            print('Class {} - AP: {}'.format(class_dict[c], self.AP[i]))
            text_file.write('Class {} - AP: {} \n'.format(class_dict[c], self.AP[i]))
            text_file.write('True Positives - {}\n False Positives - {}\n\n'.format(len(self.true_positives_list),
                                                                                    len(self.false_positives_list)))

        print('mAP: {}'.format(self.AP.mean()))
        text_file.write('mAP: {}'.format(self.AP.mean()))

        text_file.close()

        return
