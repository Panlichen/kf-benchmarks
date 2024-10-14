import os
import random
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def create_tf_example(image_path, label):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    
    # 获取图像尺寸
    image = Image.open(image_path)
    width, height = image.size

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(encoded_image),
        'image/format': _bytes_feature('jpeg'.encode()),
        'image/class/label': _int64_feature(label),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
    }))
    return tf_example

def convert_dataset(image_dir, output_path, num_shards=1024):
    class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    class_to_label = {class_name: i for i, class_name in enumerate(sorted(class_dirs))}

    image_paths = []
    labels = []

    for class_dir in class_dirs:
        class_path = os.path.join(image_dir, class_dir)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image_paths.append(image_path)
            labels.append(class_to_label[class_dir])

    # 打乱数据
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    num_images = len(image_paths)
    images_per_shard = num_images // num_shards

    for shard_id in range(num_shards):
        output_filename = f'{output_path}-{shard_id:05d}-of-{num_shards:05d}'
        start_ndx = shard_id * images_per_shard
        end_ndx = min((shard_id + 1) * images_per_shard, num_images)

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i in range(start_ndx, end_ndx):
                tf_example = create_tf_example(image_paths[i], labels[i])
                tfrecord_writer.write(tf_example.SerializeToString())

        print(f'Finished writing {output_filename}')

# 使用示例
train_dir = '/HOME/scz1075/run/data/tiny-imagenet-200/train'
val_dir = '/HOME/scz1075/run/data/tiny-imagenet-200/val'
output_dir = '/HOME/scz1075/run/data/tiny-imagenet-200/tfrecords'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 转换训练集
convert_dataset(train_dir, os.path.join(output_dir, 'train'))

# 转换验证集
convert_dataset(val_dir, os.path.join(output_dir, 'validation'))