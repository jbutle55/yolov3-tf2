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
weights = '/home/justin/Models/yolov3-tf2/checkpoints/yolov3_608.tf'
tiny = False
size = 608
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

    yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights)
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

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=0.0)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if output:
            out.write(img)
        #cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

        count = count + 1

    cv2.destroyAllWindows()


    if args.roc:
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

                print('Frame: {}'.format(count))

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
                # Detect returns:
                # "rois" []
                # "class_ids" [N]
                # "scores" [N]

                # print('Pred Image')
                # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                #                            r['scores'], figsize=(8, 8))

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
        utils.compute_roc_curve(all_tp_rates, all_fp_rates, save_fig=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video')
    parser.add_argument('--debug_model', action='store_true')
    parser.add_argument('--roi_layer', action='store_true')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    main(arguments)
