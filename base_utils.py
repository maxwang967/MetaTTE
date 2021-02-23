# -*- coding: utf-8 -*-
# @Author  : morningstarwang
# @FileName: base_utils.py
# @Blog: wangchenxing.com
import tensorflow as tf
import numpy as np


def lr_fn(n_epoch, lr_reduce, base_lr=1e-2):
    if n_epoch % 5 == 0 and n_epoch // 5 > 0:
        return base_lr * lr_reduce
    else:
        return base_lr


def masked_mae_tf(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.math.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_mse_tf(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.math.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_rmse_tf(preds, labels, null_val=np.nan):
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))


def masked_mape_tf(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.math.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.math.divide_no_nan(tf.subtract(preds, labels), labels))
    loss = loss * mask
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)
