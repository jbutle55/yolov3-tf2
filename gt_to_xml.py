import argparse
import os
import fnmatch
import xml.etree.ElementTree as ET
import re
from xml.dom import minidom
import cv2
from tools.voc2012 import parse_xml, build_example
import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm


def main(args):
    print('Parsing GT file.')
    dataset_dir = args.dataset  # Path to images of single dataset
    gt_dir = args.gt  # Path of ground truth folder

    os.chdir(dataset_dir)

    # Should be called in a sub-dataset's folder ie. M0101, M0102, etc
    if not os.path.isdir('Annotations'):
        os.mkdir('Annotations')
    os.chdir('Annotations')

    sub_id = dataset_dir[-5:]  # Image folder name ex. M0101
    print('Dataset ID: ' + sub_id)
    # Path to single ground truth file
    gt_dir = os.path.join(gt_dir, '{}_gt_whole.txt'.format(sub_id))

    frame_list = os.scandir(os.path.join(dataset_dir, 'JPEGImages'))  # Get name of all frames in dataset

    search = re.compile(r'.+(img\d+).jpg\S+')
    frames = []

    for frame in frame_list:  # Loop through each frame number of dataset
        if '.jpg' not in str(frame):
            continue

        frame_stripped = search.match(str(frame)).group(1)
        img = cv2.imread('{}/{}/{}.jpg'.format(dataset_dir, 'JPEGImages', frame_stripped))
        im_height, im_width = img.shape[:2]

        frames.append(frame_stripped)
        print('Frame: {}'.format(frame_stripped))
        annot_dict = {}  # Start with clean dict
        with open(gt_dir, 'r') as f:  # Open ground truth text file
            for line in f:  # Read each line of txt file
                gt_items = line.split(',')  # Split line into list of items
                if frame_stripped.lstrip('img0') == gt_items[0]:  # Match frame (img) numbers
                    # Each frame will contain a list of objects each with
                    # a single target id and a set of dimensions
                    target_id = gt_items[1]
                    # Append to dict for each distinct object
                    annot_dict[target_id] = {'xleft': gt_items[2],
                                             'ytop': gt_items[3],
                                             'width': gt_items[4],
                                             'height': gt_items[5],
                                             'view': gt_items[6],
                                             'occlusion': gt_items[7][:-1],
                                             'class': gt_items[8]}
        f.close()  # Close text file
        create_xml(frame_stripped, annot_dict, dataset_dir, im_width, im_height)  # Create xml for this frame

    os.chdir('..')
    with open('img_set.txt', 'w') as f:
        for img in frames:
            f.write('{}\n'.format(img))
        f.close()

    print('Done Parsing Ground Truth')

    if args.tfrecord is True:
        print('Building tfrecord file.')

        # Path to classes .names file
        classes = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/yolov3-tf2/classes.names'
        # Path to output file
        output_file = 'output.tfrecord'
        #data_dir = os.path.join(os.getcwd())
        #if not os.path.isdir(data_dir):
        data_dir = os.getcwd()  # Path to directory containing jpeg images folder
        class_map = {name: idx for idx, name in enumerate(open(classes).read().splitlines())}
        logging.info("Class mapping loaded: %s", class_map)

        writer = tf.io.TFRecordWriter(output_file)
        image_list = open(os.path.join(data_dir, 'img_set.txt')).read().splitlines()
        logging.info("Image list loaded: %d", len(image_list))
        for image in tqdm.tqdm(image_list):
            #name, _ = image.split()
            name = image
            annotation_xml = os.path.join(data_dir, 'Annotations', name + '.xml')
            annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
            annotation = parse_xml(annotation_xml)['annotation']
            tf_example = build_example(annotation, class_map)
            writer.write(tf_example.SerializeToString())
        writer.close()
        logging.info("\nDone\n")

    return


def create_xml(img_id, dictionary, img_dir, im_width, im_height):
    annot = ET.Element('annotation')

    folder = ET.SubElement(annot, 'folder')  # Image folder path
    folder.text = '{}'.format(img_dir)

    filename = ET.SubElement(annot, 'filename')  # Name of image file
    filename.text = '{}.jpg'.format(img_id)

    path = ET.SubElement(annot, 'path')  # Path of specific image file
    path.text = os.path.join(img_dir, '{}.jpg'.format(img_id))

    source = ET.SubElement(annot, 'source')
    database = ET.SubElement(source, 'database')  # Database name
    database.text = 'uav'

    size = ET.SubElement(annot, 'size')  # Contains size of image
    width = ET.SubElement(size, 'width')
    width.text = str(im_width)
    height = ET.SubElement(size, 'height')
    height.text = str(im_height)
    depth = ET.SubElement(size, 'depth')  # 1 for B&W, 3 for colour
    depth.text = str(3)

    segment = ET.SubElement(annot, 'segmented')

    obj_class_dict = {'1': 'car', '3': 'bus', '5': 'motorbike', '2': 'truck'}

    w, h, d = None, None, None

    for object in dictionary.keys():  # Iteration of all keys (individual objects)
        #if dictionary[object].get('class').rstrip() not in obj_class_dict.keys():
        #    continue

        # Object insertion for each individual object
        obj = ET.SubElement(annot, 'object')
        name = ET.SubElement(obj, 'name')
        if dictionary[object].get('class').rstrip() not in obj_class_dict.keys():
            name.text = dictionary[object].get('class').rstrip()
            print(dictionary[object].get('class').rstrip())
        else:
            name.text = obj_class_dict[dictionary[object].get('class').rstrip()]  # Object id
        # name.text = obj_class_dict[class_label]  # Object id

        pose = ET.SubElement(obj, 'pose')
        trunc = ET.SubElement(obj, 'truncated')
        diff = ET.SubElement(obj, 'difficult')

        x_left = dictionary[object].get('xleft')
        y_top = dictionary[object].get('ytop')
        w = dictionary[object].get('width')
        h = dictionary[object].get('height')

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = x_left

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(y_top))

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(x_left) + int(w))

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(y_top) + int(h))

        d = dictionary[object].get('depth')

    # Create file
    tree = ET.ElementTree(annot)
    
    # Prettify
    rough_string = ET.tostring(annot, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent='  ')
    
    string = '{}.xml'.format(img_id)
    with open(string, 'w') as f:
        f.write(pretty)
    f.close()
    
    #pretty.write(string)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', help='path to folder of images for single dataset')
    parser.add_argument('--gt', default='/Users/justinbutler/Desktop/school/Calgary/Thesis Work/Datasets/UAV-benchmark-M/UAV-benchmark-MOTD_v1.0/GT', help='path to gt file')
    parser.add_argument('--tfrecord', default=False,
                        action='store_true', help='Boolian whether to create tfrecord file or not. Default False')
    args = parser.parse_args()
    main(args)
