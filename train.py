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

print(tf.__version__)
print(sys.version)


def main(args):
    train_path = args.train_dataset
    valid_path = args.valid_dataset
    weights_path = args.weights
    # Path to text? file containing all classes, 1 per line
    classes = args.classes
    # Usually fit
    #mode = 'fit'  # Can be 'fit', 'eager_fit', 'eager_tf', 'valid'
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
    image_size = 416
    #num_epochs = 3
    num_epochs = args.epochs
    #batch_size = 16
    batch_size = args.batch_size
    learning_rate = 1e-3
    #num_classes = 9
    num_classes = args.num_classes
    # num class for `weights` file if different, useful in transfer learning with different number of classes
    #weight_num_classes = 80
    weight_num_classes = args.num_weight_class
    
    saved_weights_path = '/weights/trained_weights'

    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if args.no_train:
        print('Skipping training...')
    else:
        model = YoloV3(image_size, training=True, classes=num_classes)

        train_dataset = dataset.load_tfrecord_dataset(train_path,
                                                      classes,
                                                      image_size)
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                    classes,
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
                ReduceLROnPlateau(verbose=1),
                # EarlyStopping(patience=3, verbose=1),
                ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                                verbose=1, save_weights_only=True),
                TensorBoard(log_dir='logs')
            ]

            history = model.fit(train_dataset,
                                epochs=num_epochs,
                                callbacks=callbacks,
                                validation_data=val_dataset)

            model.save_weights(saved_weights_path)

    # Calculate mAP
    if args.validate:
        model = YoloV3(image_size, training=False, classes=num_classes)
        model.load_weights(saved_weights_path).expect_partial()

        batch_size = 1

        val_dataset = dataset.load_tfrecord_dataset(valid_path,
                                                    classes,
                                                    image_size)
        val_dataset = val_dataset.batch(batch_size)

        val_dataset = val_dataset.map(lambda x, y: (
            dataset.transform_images(x, image_size),
            dataset.transform_targets(y, anchors, anchor_masks, image_size)))

        predictions = []

        evaluator = Evaluator(iou_thresh=0.5)

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
        #for img in range(len(full_labels_flat)):
        for img in full_labels_flat:
            test = []
            #for scale in full_labels_flat[img]:
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
        #filt_labels = filt_labels[:, :4] * image_size

        # i is the num_images index
        #predictions = [np.hstack([boxes[i][x], scores[i][x], classes[i][x]]) for i in range(len(num_detections)) for x in range(len(scores[i])) if scores[i][x] > 0]
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

        #predictions[:, :, 0:4] = predictions[:, :, 0:4] * image_size
        # Predictions format - [num_imgs x num_preds x [box coords x4, score, classes]]
        # Box coords should be in format x1 y1 x2 y2

        evaluator(predictions, filt_labels)  # Check gts box coords

    mode = 'valid'
    if mode == 'valid':
        print('Validation Image...')
        # Find better way to do this so not requiring manual changes
        class_dict = {0: 'car', 1: 'bus', 2: 'truck', 3: 'person',
                      4: 'bicycle', 5: 'motorbike', 6: 'train',
                      7: 'building', 8: 'traffic light'}
        output = '/Users/justinbutler/Desktop/test/tiny_test/test.jpg'  # Path to output image

        class_names = list(class_dict.values())
        print('classes loaded')

        #val_dataset = dataset.load_tfrecord_dataset(valid_path,
        #                                            classes,
        #                                            image_size)
        #val_dataset = val_dataset.map(lambda x, y: (
        #    dataset.transform_images(x, image_size),
        #    dataset.transform_targets(y, anchors, anchor_masks, image_size)))

        test2 = []
        index = 0
        for img in val_dataset:
            test2.append(img)

        # boxes, scores, classes, num_detections
        test = []
        index = 0
        for img in val_dataset:
            image = img[0][0]
            output = '/Users/justinbutler/Desktop/test/tiny_test/test_{}.jpg'.format(index)

            # image has values between 0 and 1
            image = image * 255

            print('detections:')
            for i in range(num_detections[index]):
                print('\t{}, {}, {}'.format(class_names[int(classes[index][i])],
                                            np.array(scores[index][i]),
                                            np.array(boxes[index][i])))

            img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes[index], scores[index], classes[index], num_detections[index]), class_names)
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

    args = parser.parse_args()
    main(args)
