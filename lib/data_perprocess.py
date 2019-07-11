"""
Name : data_perprocess.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-07-10 18:25
Desc:
"""
import numpy as np


def data_perprocess(data, operation='sub'):
    """

    :param data:dtype = np.float32
    :param operation:
    :return:
    """
    assert data.dtype == np.float32
    if operation == "sub":
        return subtract_mean(data)
    elif operation == "divnorm":
        return div_norm(data)


def subtract_mean(data):
    """
    subtract mean
    :param data:
    :return:
    """
    assert len(data.shape) > 3
    if len(data.shape) == 3:
        data[:, :, 0] = data[:, :, 0] - 103.939
        data[:, :, 1] = data[:, :, 1] - 116.779
        data[:, :, 2] = data[:, :, 2] - 123.68
    elif len(data.shape) == 4:
        data[:, :, :, 0] = data[:, :, :, 0] - 103.939
        data[:, :, :, 1] = data[:, :, :, 1] - 116.779
        data[:, :, :, 2] = data[:, :, :, 2] - 123.68
    else:
        print("bad shape")
    return data


def div_norm(data):
    """
    div_norm
    :param data:
    :return:
    """
    assert len(data.shape) > 3
    if len(data.shape) == 3:
        data[:, :, 0] = data[:, :, 0:2] / data[:, :, 0:2].max()
    elif len(data.shape) == 4:
        data[:, :, :, 0:2] = data[:, :, :, 0:2] / data[:, :, 0:2].max()
    else:
        print("bad shape")
    return data
