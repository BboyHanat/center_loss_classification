"""
Name : run.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-11 11:19
Desc:
"""
from trainer.train_center_loss import train

train_img_root = "./dataset/train_fonts99/train"
val_img_root = "./dataset/train_fonts99/val"
height = 512
width = 512
channels = 3
class_num = 2
batch_size = 16
epoch = 64


train(train_img_root, val_img_root, height=224, width=224, channels=3, class_num=109, train_batch_size=32, val_batch_size=16, epoch=64)


# import tensorflow as tf
#
# limit = tf.placeholder(dtype=tf.int32, shape=[])
#
# dataset = tf.data.Dataset.from_tensor_slices((tf.range(start=0, limit=limit),tf.range(start=0, limit=limit)))
#
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={limit: 10})
#     for i in range(10):
#       value1, value2 = sess.run(next_element)
#       print(value1,value2)
#       assert i == value1