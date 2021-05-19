import tensorflow as tf
import sys
import numpy as np
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from yolov3_tf2.models import YoloV3, YoloV3Tiny, YoloLoss, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
from yolov3_tf2.utils import freeze_all, draw_outputs
import yolov3_tf2.dataset as dataset
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import time
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from eval_utils import Evaluator
import argparse
import os
import time
import config as cfg

print(tf.__version__)
print(sys.version)


# Flatten all labels and strip labels containing all 0's
def flatten_labels(label):
    filt_labels = []

    # Flatten all labels and remove 0s
    for size in label:
        for grid1 in size:
            for grid2 in grid1:
                for anchor in grid2:
                    for a in anchor:
                        if a[4] > 0:
                            temp = [a[0] * cfg.IMAGE_SIZE,
                                    a[1] * cfg.IMAGE_SIZE,
                                    a[2] * cfg.IMAGE_SIZE,
                                    a[3] * cfg.IMAGE_SIZE,
                                    a[4],
                                    a[5]]
                            temp = [float(x) for x in temp]
                            filt_labels.append(np.asarray(temp))
    filt_labels = np.asarray(filt_labels)

    return filt_labels


def main(args):
    tf.config.experimental.list_physical_devices('GPU')
    # tf.device(f'/gpu:{args.gpu_num}')

    train_path = args.train_dataset
    valid_path = args.valid_dataset
    weights_path = args.weights
    # Path to text? file containing all classes, 1 per line
    classes_file = args.classes
    # Usually fit
    # mode = 'fit'  # Can be 'fit', 'eager_fit', 'eager_tf', 'valid'
    mode = args.mode
    '''
    'fit: model.fit, '
    'eager_fit: model.fit(run_eagerly=True), '
    'eager_tf: custom GradientTape'
    '''

    # Usually darknet
    transfer = args.transfer
    '''
    'none: Training from scratch, '
                      'darknet: Transfer darknet, '
                      'no_output: Transfer all but output, '
                      'frozen: Transfer and freeze all, '
                      'fine_tune: Transfer all and freeze darknet only'),
                      'pre': Use a pre-trained model for validation
    '''
    image_size = cfg.IMAGE_SIZE

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = cfg.LEARNING_RATE
    num_classes = args.num_classes
    # num class for `weights` file if different, useful in transfer learning with different number of classes
    weight_num_classes = args.num_weight_class

    # saved_weights_path = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/yolov3-tf2/weights/'
    saved_weights_path = '/home/justin/ml_models/yolov3-tf2/weights/trained_{}.tf'.format(num_epochs)
    saved_weights_path = args.saved_weights

    # Original Anchors below
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / 608

    anchors = cfg.YOLO_ANCHORS

    anchor_masks = cfg.YOLO_ANCHOR_MASKS

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if args.no_train:
        print('Skipping training...')
    else:
        start_time = time.time()
        model = YoloV3(image_size, training=True, classes=num_classes)

        train_dataset = dataset.load_tfrecord_dataset(train_path,
                                                      classes_file,
                                                      image_size)
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                    classes_file,
                                                    image_size)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))

        # Configure the model for transfer learning
        if transfer == 'none':
            pass  # Nothing to do
        elif transfer in ['darknet', 'no_output']:
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

            elif transfer == 'no_output':
                for layer in model.layers:
                    if not layer.name.startswith('yolo_output'):
                        layer.set_weights(model_pretrained.get_layer(
                            layer.name).get_weights())
                        freeze_all(layer)
        elif transfer == 'pre':
            model = YoloV3(image_size,
                           training=False,
                           classes=num_classes)
            model.load_weights(weights_path)

        else:
            # All other transfer require matching classes
            model.load_weights(weights_path)
            if transfer == 'fine_tune':
                # freeze darknet and fine tune other layers
                darknet = model.get_layer('yolo_darknet')
                freeze_all(darknet)
            elif transfer == 'frozen':
                # freeze everything
                freeze_all(model)
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        loss = [YoloLoss(anchors[mask], classes=num_classes)
                for mask in anchor_masks]  # Passing loss as a list might sometimes fail? dict might be better?

        if mode == 'eager_tf':
            # Eager mode is great for debugging
            # Non eager graph mode is recommended for real training
            avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
            for epoch in range(1, num_epochs + 1):
                for batch, (images, labels) in enumerate(train_dataset):
                    with tf.GradientTape() as tape:
                        outputs = model(images, training=True)
                        regularization_loss = tf.reduce_sum(model.losses)
                        pred_loss = []
                        for output, label, loss_fn in zip(outputs, labels, loss):
                            pred_loss.append(loss_fn(label, output))
                        total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables))
                    print("{}_train_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_loss.update_state(total_loss)
                for batch, (images, labels) in enumerate(val_dataset):
                    outputs = model(images)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                    print("{}_val_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                    avg_val_loss.update_state(total_loss)
                print("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))
                avg_loss.reset_states()
                avg_val_loss.reset_states()

                model.save_weights(
                    'checkpoints/yolov3_train_{}.tf'.format(epoch))
        elif mode == 'valid':
            pass  # Pass this step for validation only
        else:
            model.compile(optimizer=optimizer, loss=loss,
                          run_eagerly=(mode == 'eager_fit'))
            callbacks = [
                ReduceLROnPlateau(verbose=1, min_lr=1e-4, patience=50),
                # EarlyStopping(patience=3, verbose=1),
                ModelCheckpoint('checkpoints/midpoints/yolov3_train_{epoch}.tf',
                                verbose=1, save_weights_only=True),
                TensorBoard(log_dir=f'logs/{saved_weights_path[:-3]}')
            ]

            history = model.fit(train_dataset,
                                epochs=num_epochs,
                                callbacks=callbacks,
                                validation_data=val_dataset)
            print(f'Saving weights to: {saved_weights_path}')
            model.save_weights(saved_weights_path)
        finish_time = time.time()
        train_time = finish_time - start_time
        print('Training time elapsed: {}'.format(train_time))

    # Calculate mAP
    if args.validate:
        print('Validating...')
        model = YoloV3(image_size, training=False, classes=num_classes)
        model.load_weights(saved_weights_path).expect_partial()

        batch_size = 1

        val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                    classes_file,
                                                    image_size)
        val_dataset = val_dataset.batch(batch_size)

        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))

        images = []
        for img, labs in val_dataset:
            img = np.squeeze(img)
            images.append(img)

        predictions = []

        evaluator = Evaluator(iou_thresh=args.iou)

        # labels - (N, grid, grid, anchors, [x, y, w, h, obj, class])
        boxes, scores, classes, num_detections = model.predict(val_dataset)
        # boxes -> (num_imgs, num_detections, box coords)

        filtered_labels = []
        for _, label in val_dataset:
            filt_labels = flatten_labels(label)
            filtered_labels.append(filt_labels)

        # i is the num_images index
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

        evaluator(predictions, filtered_labels, images, roc=False)  # Check gts box coords

    if args.valid_imgs:  # Predictions
        print('Valid Images...')
        # yolo = YoloV3(classes=num_classes)
        yolo = YoloV3(image_size, training=False, classes=num_classes)
        yolo.load_weights(saved_weights_path).expect_partial()
        print('weights loaded')

        print('Validation Image...')
        # Find better way to do this so not requiring manual changes
        class_dict = cfg.CLASS_DICT

        class_names = list(class_dict.values())
        print('classes loaded')

        val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                    classes_file,
                                                    image_size)
        val_dataset = val_dataset.batch(1)
        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))


        # boxes, scores, classes, num_detections
        index = 0
        for img_raw, _label in val_dataset.take(5):
            print(f'Index {index}')

            #img = tf.expand_dims(img_raw, 0)
            img = transform_images(img_raw, image_size)
            img = img * 255

            boxes, scores, classes, nums = yolo(img)

            filt_labels = flatten_labels(_label)

            boxes = tf.expand_dims(filt_labels[:, 0:4], 0)
            scores = tf.expand_dims(filt_labels[:, 4], 0)
            classes = tf.expand_dims(filt_labels[:, 5], 0)
            nums = tf.expand_dims(filt_labels.shape[0], 0)

            img = cv2.cvtColor(img_raw[0].numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=0)
            # img = img * 255

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

    if args.visual_data:
        print('Visual Data...')
        val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                    classes_file,
                                                    image_size)
        val_dataset = val_dataset.batch(1)
        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))

        index = 0
        for img_raw, _label in val_dataset.take(5):
            print(f'Index {index}')
            # img = tf.expand_dims(img_raw, 0)
            img = transform_images(img_raw, image_size)

            output = 'test_images/test_labels_{}.jpg'.format(index)
            # output = '/Users/justinbutler/Desktop/test/test_images/test_labels_{}.jpg'.format(index)

            filt_labels = flatten_labels(_label)

            boxes = tf.expand_dims(filt_labels[:, 0:4], 0)
            scores = tf.expand_dims(filt_labels[:, 4], 0)
            classes = tf.expand_dims(filt_labels[:, 5], 0)
            nums = tf.expand_dims(filt_labels.shape[0], 0)

            img = cv2.cvtColor(img_raw[0].numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names, thresh=0)
            img = img * 255

            cv2.imwrite(output, img)

            index = index + 1

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_dataset', help='')
    parser.add_argument('--valid_dataset', help='')
    parser.add_argument('--weights', help='')
    parser.add_argument('--classes', help='')
    parser.add_argument('--num_classes', type=int, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--mode', default='fit', help='')
    parser.add_argument('--transfer', default='darknet', required=False, help='')
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='')
    parser.add_argument('--num_weight_class', type=int, default=80, required=False, help='')
    parser.add_argument('--no_train', '-nt', action='store_true', help='')
    parser.add_argument('--validate', '-v', action='store_true', help='')
    parser.add_argument('--valid_imgs', action='store_true', default=False)
    parser.add_argument('--iou', required=False, default=0.5, type=float)
    parser.add_argument('--saved_weights', default='/weights/trained_model.tf',
                        help='Also the model path for validation if running with no training.')
    parser.add_argument('--output_dir', help='')
    parser.add_argument('--visual_data', action='store_true', default=False)
    parser.add_argument('--gpu_num', default=0)

    args = parser.parse_args()
    main(args)
