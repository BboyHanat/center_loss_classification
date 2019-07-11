"""
Name : train_center_loss.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-11 10:16
Desc:
"""

from network import NetWork
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import random
import numpy as np


def get_data_shuffle(class_root, train=True):
    """

    :param class_root:
    :param train:
    :return:
    """

    fp = open("labels.txt", 'w+')
    class_path = [os.path.join(class_root, img_path) for img_path in os.listdir(class_root)
                  if os.path.isdir(os.path.join(class_root, img_path))]
    class_path.sort()
    imgs_path = []
    labels = []
    for index, img_class in enumerate(class_path):
        if train:
            str_index = str(index)
            line = str_index + ":" + img_class.split('/')[-1] + '\n'
            fp.write(line)

        image_paths = [os.path.join(img_class, image_path) for image_path in os.listdir(img_class)
                       if image_path.split('.')[-1] in ['jpg', 'png', 'JPEG', 'jpeg', 'PNG', 'bmp']
                       and not image_path.startswith('.DS')]
        for i in range(len(image_paths)):
            labels.append(index)
        imgs_path += image_paths

    assert len(imgs_path) == len(labels)
    merge_list = [[img, label] for img, label in zip(imgs_path, labels)]
    random.shuffle(merge_list)
    imgs_path = []
    labels = []
    merge_list = tuple(merge_list)
    for image, label in merge_list:
        imgs_path.append(image)
        labels.append(label)
    return imgs_path, labels


def train(train_img_root,
          val_img_root,
          height=512,
          width=512,
          channels=3,
          class_num=109,
          train_batch_size=16,
          val_batch_size=16,
          epoch=64
          ):
    """

    :param train_img_root:
    :param val_img_root:
    :param height:
    :param width:
    :param channels:
    :param class_num:
    :param batch_size:
    :param epoch:
    :return:
    """

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_resized = tf.image.resize_image_with_pad(image_decoded, height, width)
        label = tf.cast(label, tf.float32)
        return image_resized, label

    """get train and validation files path with labels,respectively"""
    train_images_path, train_labels = get_data_shuffle(train_img_root)
    training_iters = len(train_images_path) // train_batch_size
    val_image_path, val_labels = get_data_shuffle(val_img_root, train=False)
    """convert list to tensor"""
    train_images_path = tf.constant(train_images_path)
    train_labels = tf.constant(train_labels)
    val_image_path = tf.constant(val_image_path)
    val_labels = tf.constant(val_labels)
    """get training iterations"""

    sess = tf.Session()
    """train data"""
    dataset_train = tf.data.Dataset.from_tensor_slices((train_images_path, train_labels))
    dataset_train = dataset_train.map(_parse_function)
    dataset_train = dataset_train.shuffle(buffer_size=1000).batch(train_batch_size).repeat(epoch)

    """validation data"""
    dataset_val = tf.data.Dataset.from_tensor_slices((val_image_path, val_labels))
    dataset_val = dataset_val.map(_parse_function)
    dataset_val = dataset_val.shuffle(buffer_size=1000).batch(val_batch_size).repeat()

    sess.run(tf.local_variables_initializer())

    net = NetWork(sess=sess,
                  backbones='resnet_v1_50',
                  pretrained_model='weight/pretrain/resnet_v1_50.ckpt',
                  width=width,
                  height=height,
                  channels=channels,
                  class_num=class_num
                  )
    net.train(dataset_train=dataset_train,
              dataset_val=dataset_val,
              epochs=epoch,
              training_iters=training_iters,
              ckpt_path='./weight/train/'
              )
    sess.close()
