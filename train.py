import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from yolov3_tf2.models import YoloV3, YoloV3Tiny, YoloLoss, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

train_path = '/Users/justinbutler/Desktop/school/Calgary/Thesis Work/ML_Testing/yolov3-tf2/train/JPEGImages/*.jpg'
valid_path = '/Users/justinbutler/Desktop/school/Calgary/Thesis Work/ML_Testing/yolov3-tf2/data/voc2012_train.tfrecord'
weights_path = './checkpoints/yolov3.weights'
classes = '/Users/justinbutler/Desktop/school/Calgary/Thesis Work/ML_Testing/yolov3-tf2/data/voc2012.names'  # Path to text? file containing all classes, 1 per line
mode = 'eager_tf'  # Can be 'fit', 'eager_fit', 'eager_tf'
'''
'fit: model.fit, '
'eager_fit: model.fit(run_eagerly=True), '
'eager_tf: custom GradientTape'
'''

transfer = 'darknet'  # Can be 'none', 'darknet', 'no_output', 'frozen', 'fine_tune'
'''
'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
'''
image_size = 416
num_epochs = 1
batch_size = 8
learning_rate = 1e-3
num_classes = 3
weight_num_classes = 80  # num class for `weights` file if different, useful in transfer learning with different number of classes

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

model = YoloV3(image_size, training=True, classes=num_classes)
anchors = yolo_anchors
anchor_masks = yolo_anchor_masks

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
        for mask in anchor_masks]

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
            logging.info("{}_train_{}, {}, {}".format(
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
            logging.info("{}_val_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_val_loss.update_state(total_loss)
        logging.info("{}, train: {}, val: {}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy()))
        avg_loss.reset_states()
        avg_val_loss.reset_states()

        model.save_weights(
            'checkpoints/yolov3_train_{}.tf'.format(epoch))
else:
    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(mode == 'eager_fit'))
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(train_dataset,
                        epochs=num_epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)
