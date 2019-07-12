"""
Name : tools.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-10 18:00
Desc:
"""

import tensorflow as tf



def binary_tf_confusion_metrics(predict, labels):
    """
    binary_tf_confusion_metrics
    :param predict:
    :param labels:
    :return:
    """
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(labels, 1)
    ones_like_actuals = tf.ones_like(
        actuals)  # 维度和actuals一样的全1的张量   
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals), tf.equal(predictions, zeros_like_predictions)), "float"))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, zeros_like_predictions)), "float"))
    return tp, fn, fp, tn


def binary_f1_score(predict, labels):
    """
    f1_score
    :param predict:
    :param labels:
    :return:
    """
    tp, fn, fp, tn = binary_tf_confusion_metrics(predict, labels)
    precision = float(tp) / (float(tp) + float(fp))
    recall = float(tp) / (float(tp) + float(fn))
    return (2 * (precision * recall)) / (precision + recall)


def binary_accuary(predict, labels):
    """
    accuary
    :param predict:
    :param labels:
    :return:
    """
    tp, fn, fp, tn = binary_tf_confusion_metrics(predict, labels)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return accuracy


def binary_recall(predict, labels):
    """
    recall
    :param predict:
    :param labels:
    :return:
    """
    tp, fn, fp, tn = binary_tf_confusion_metrics(predict, labels)
    return float(tp) / (float(tp) + float(fn))


def binary_precision(predict, labels):
    """
    Precision
    :param predict:
    :param labels:
    :return:
    """
    tp, fn, fp, tn = binary_tf_confusion_metrics(predict, labels)
    return tp / (tp + fp)


