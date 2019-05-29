from config import *

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf

def tf_get_crop_value(img):

    img_w = tf.cast(tf.shape(img)[1],dtype=tf.float32)
    img_h = tf.cast(tf.shape(img)[0],dtype=tf.float32)

    x1_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_w*0.05),dtype=tf.int32),[])
    x2_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_w*0.05),dtype=tf.int32),[])
    y1_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_h*0.05),dtype=tf.int32),[])
    y2_c = tf.reshape(tf.cast(tf.random_uniform([1], minval=0, maxval=img_h*0.05),dtype=tf.int32),[])

    return [x1_c, x2_c, y1_c, y2_c]


def tf_crop(img, mask):
    img_w = tf.cast(tf.shape(img)[1],dtype=tf.int32)
    img_h = tf.cast(tf.shape(img)[0],dtype=tf.int32)
    x1_c, x2_c, y1_c, y2_c = tf_get_crop_value(img)

    img = img[y1_c:img_h-y2_c, x1_c:img_w-x2_c]
    mask = mask[y1_c:img_h-y2_c, x1_c:img_w-x2_c]

    return img, mask


def parse_func(serialized_example):
    features = tf.parse_single_example(serialized_example, {'shape': tf.FixedLenFeature([3], tf.int64),
                                                            'data': tf.FixedLenFeature([], tf.string),
                                                            'mask': tf.FixedLenFeature([], tf.string)})

    img = tf.reshape(tf.decode_raw(features['data'], tf.uint8), [3, loading_img_h, loading_img_w])
    img = tf.transpose(img, [1, 2, 0])

    mask = tf.reshape(tf.decode_raw(features['mask'], tf.uint8), [loading_img_h, loading_img_w, 1])

    #cast to proper type
    img = tf.cast(img, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # normalize image from 0 - 1
    img = (img-255/2)/(255/2)

    img, mask = tf_crop(img, mask)

    #downsample to proper size for depeneds on phase
    img = tf.image.resize_images(img, [8 * 2 ** (phase - 1), 8 * 2 ** (phase - 1)])
    mask = tf.image.resize_images(mask, [8 * 2 ** (phase - 1), 8 * 2 ** (phase - 1)])
    mask = tf.cast(mask > 0, tf.float32)

    img = tf.reshape(img, [8 * 2 ** (phase - 1), 8 * 2 ** (phase - 1), 3])
    mask = tf.reshape(mask, [8 * 2 ** (phase - 1), 8 * 2 ** (phase - 1), 1])

    img, mask = flip(img, mask)
    return img, mask



def flip(img, mask):
    do_flip = tf.reshape(tf.random_uniform([1], minval=0, maxval=1),[]) > tf.reshape(FLIP_RATE,[])
    img = tf.cond(do_flip, lambda:tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(do_flip, lambda:tf.image.flip_left_right(mask), lambda: mask)

    return img, mask


def build_input_pipline(batch_size, train_filename):
    ds_train = tf.data.TFRecordDataset(train_filename)
    ds_train = ds_train.map(parse_func, num_parallel_calls=48)

    ds_train = ds_train.shuffle(500).repeat().batch(batch_size).prefetch(batch_size * 3) # add shuffling
    iterator_train = ds_train.make_one_shot_iterator()


    return iterator_train.get_next()


if __name__ == '__main__':
    img, mask = build_input_pipline(batch_size, train_tfrecord_path)
    config = tf.ConfigProto(allow_soft_placement=True)
    plt.ion()

    with tf.Session(config=config) as sess:
        for _ in tqdm(range(10000)):
            img_l, mask_l = sess.run([img, mask])
            print(img_l.shape, mask_l.shape)

            img_l = img_l[0,:,:,:]
            mask_l = mask_l[0,:,:,0]
            img_l = img_l * 255 / 2 + 255 / 2

            mask_l[mask_l>0] = 1
            plt.figure(1)
            plt.imshow(mask_l)
            plt.figure(2)
            plt.imshow(img_l.astype(np.uint8))
            plt.pause(3)

