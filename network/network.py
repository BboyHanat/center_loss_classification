"""
Name : network.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-10 15:05
Desc:
"""

from tensorflow.contrib import slim
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from lib.resnet_v1 import *
from lib.center_loss import *
from lib.data_perprocess import *
from lib.measurement_tools import *
from operator import methodcaller
import os
import collections


class NetWork:
    """
    cnn model
    """

    _networks_map = {
        'resnet_v1_50': {'function': resnet_v1_50,
                         'C1': 'resnet_v1_50/conv1',
                         'C2': 'resnet_v1_50/block1/unit_2/bottleneck_v1',
                         'C3': 'resnet_v1_50/block2/unit_3/bottleneck_v1',
                         'C4': 'resnet_v1_50/block3/unit_5/bottleneck_v1',
                         'C5': 'resnet_v1_50/block4/unit_3/bottleneck_v1',
                         'logit': 'resnet_v1_50/logits_fc',
                         'data_process_op': 'sub'
                         },
        'resnet_v1_101': {'function': resnet_v1_101,
                          'C1': '',
                          'C2': '',
                          'C3': '',
                          'C4': '',
                          'C5': '',
                          'data_process_op': 'sub'
                          },
        'inception_v3': {'function': None,
                         'C1': '',
                         'C2': '',
                         'C3': '',
                         'C4': '',
                         'C5': '',
                         'data_process_op': 'sub'
                         },
        'inception_resnet_v2': {'function': None,
                                'C1': '',
                                'C2': '',
                                'C3': '',
                                'C4': '',
                                'C5': '',
                                'data_process_op': 'sub'
                                }
    }

    def __init__(self, sess,
                 backbones,
                 pretrained_model,
                 height=None,
                 width=None,
                 channels=3,
                 class_num=109):
        """

        :param sess:
        :param backbones:
        :param pretrained_model:
        :param height:
        :param width:
        :param channels:
        :param class_num:
        """
        self.sess = sess
        self.images = tf.placeholder(tf.float32, shape=[None, 512, 512, channels])
        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.width = width
        self.height = height
        self.backbones = backbones
        self.class_num = class_num
        self.pretrained_model = pretrained_model
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step_unet")
        self.network_info, self.optimizer, self.loss, self.output, self.acc \
            = self.graph(self.global_step, learning_rate=0.001, decay_rate=0.95)
        if pretrained_model is not None:
            self.load_pretrained_model()

    def graph(self, global_step, learning_rate=0.001, decay_rate=0.95):
        """
        create a graph
        :param global_step:
        :param training_iters:
        :param learning_rate:
        :param decay_rate:
        :param momentum:
        :return:
        """
        network_info = self._networks_map[self.backbones]
        graph_func = network_info['function']
        net, end_points = graph_func(self.images, num_classes=self.class_num)
        logit = end_points[network_info["logit"]]
        logit = tf.squeeze(logit, axis=[1, 2])
        #center_loss, centers, centers_update_op = get_center_loss(logit, self.labels, 0.5, self.class_num)
        one_hot = tf.one_hot(self.labels, self.class_num)
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=logit)

        #center_loss = tf.reduce_mean(center_loss)
        softmax_loss = tf.reduce_mean(softmax_loss)
        total_loss = softmax_loss #+ 0.5 * center_loss

        acc = tf.metrics.accuracy(labels=one_hot, predictions=logit)
        #with tf.control_dependencies([centers_update_op]):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step)
        return network_info, optimizer, total_loss, logit, acc

    def load_pretrained_model(self):
        """
        load pretrained network weight
        :param sess:
        :return:
        """

        def get_variables_in_checkpoint_file(file_name):
            try:
                reader = pywrap_tensorflow.NewCheckpointReader(file_name)
                var_to_shape_map = reader.get_variable_to_shape_map()
                return var_to_shape_map
            except Exception as e:  # pylint: disable=broad-except
                print(str(e))
                if "corrupted compressed block contents" in str(e):
                    print("It's likely that your checkpoint file has been compressed "
                          "with SNAPPY.")

        def get_variables_to_restore(variables, var_keep_dic):
            variables_to_restore = []
            for v in variables:
                # exclude
                if v.name.split(':')[0] in var_keep_dic:
                    print('Variables restored: %s' % v.name)
                    variables_to_restore.append(v)
                else:
                    print('Variables restored: %s' % v.name)
            return variables_to_restore

        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables, name='init'))
        print("variables initilized ok")
        # Get dictionary of model variable
        var_keep_dic = get_variables_in_checkpoint_file(self.pretrained_model)
        # # Get the variables to restore
        variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, self.pretrained_model)

    def train(self, dataset_train, dataset_val, epochs, training_iters, val_interval=1000, val_iters=100, show_step=50, ckpt_path="./weight"):
        """
        train
        :param image:
        :param label:
        :param epochs:
        :param training_iters:
        :param ckpt_path:
        :return:
        """
        print("Start Training")
        iterator_train = dataset_train.make_initializable_iterator()
        init_op_train = iterator_train.make_initializer(dataset_train)
        iterator_val = dataset_val.make_initializable_iterator()
        init_op_val = iterator_val.make_initializer(dataset_val)
        self.sess.run([init_op_train, init_op_val])

        iterator_train = iterator_train.get_next()
        iterator_val = iterator_val.get_next()

        saver = tf.train.Saver(max_to_keep=4)
        coord = tf.train.Coordinator()
        for epoch in range(epochs):
            for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                batch_x, batch_y = self.sess.run(iterator_train)
                batch_x = data_perprocess(batch_x, self.network_info['data_process_op'])
                loss = self.sess.run([self.loss], feed_dict={self.images: batch_x,
                                                             self.labels: batch_y
                                                             })
                self.sess.run([self.optimizer, self.global_step], feed_dict={self.images: batch_x,
                                                                             self.labels: batch_y,
                                                                             })

                # validation on training
                if step % val_interval == 0 and step > val_interval:
                    accuarys = 0.0
                    for i in range(val_iters):
                        val_batch_x, val_batch_y = self.sess.run(iterator_val)
                        val_batch_x = data_perprocess(val_batch_x, self.network_info['data_process_op'])
                        acc = self.sess.run([self.acc], feed_dict={self.images: val_batch_x, self.labels: val_batch_y})
                        accuarys += acc
                    print("Accuary: {}".format(accuarys / val_iters))
                if step % show_step == 0 and step > 0:
                    print("Loss: {}".format(loss))
            ckpt_name = self.backbones + '_center_loss_' + str(epoch) + '.ckpt'
            saver.save(self.sess, os.path.join(ckpt_path, ckpt_name))
        coord.request_stop()

    def forward(self, image):
        image = data_perprocess(image, self.network_info['data_process_op'])
        image = np.expand_dims(image, axis=0)
        output = self.sess.run([self.output], feed_dict={self.images: image})
        labels = np.argsort(output)[:, ::-1]
        labels = labels[0, 0:10]
        print(labels)
        return labels
