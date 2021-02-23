# -*- coding: utf-8 -*-
# @Author  : morningstarwang
# @FileName: datasets.py
# @Blog: wangchenxing.com
import tensorflow as tf
import numpy as np


class MyDataset:
    def __init__(self, dataset, batch_size):
        lats = dataset["lats"]
        lngs = dataset["lngs"]
        labels = dataset["labels"]
        # cut to fit batch_size
        self.lats = lats[:len(lats) - (len(lats) % batch_size)]
        self.lngs = lngs[:len(lngs) - (len(lngs) % batch_size)]
        self.labels = labels[:len(labels) - (len(labels) % batch_size)]
        # set ptr
        self.current_ptr = 0
        self.is_done = False
        self.batch_num = len(self.lats) // batch_size
        self.batch_size = batch_size

    def batch(self):
        if not self.is_done:
            return self.get_batch()
        else:
            self.current_ptr = 0
            self.is_done = True
            return self.get_batch()

    def get_batch(self):
        lats = self.lats[self.current_ptr: self.current_ptr + self.batch_size]
        lngs = self.lngs[self.current_ptr: self.current_ptr + self.batch_size]
        features = [
            np.stack([lats[i], lngs[i]], axis=1) for i in range(len(lats))
        ]
        labels = self.labels[self.current_ptr: self.current_ptr + self.batch_size]
        features = [tf.convert_to_tensor(x, dtype=tf.float32) for x in features]
        features_tf = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        features_tf = features_tf.unstack(features)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return_data = features_tf, labels
        self.current_ptr += self.batch_size
        if self.current_ptr >= len(self.lats):
            self.is_done = True
        return return_data


class MyDifferDataset:
    def __init__(self, dataset, batch_size):
        lats = dataset["lats"]
        lngs = dataset["lngs"]
        labels = dataset["labels"]
        lats_differ = []
        lngs_differ = []
        # calculate differs
        for lat in lats:
            l1 = lat[:-1]
            l2 = lat[1:]
            l_differ = [abs(l1[i] - l2[i]) for i in range(len(l1))]
            lats_differ.append(l_differ)
        for lng in lngs:
            l1 = lng[:-1]
            l2 = lng[1:]
            l_differ = [abs(l1[i] - l2[i]) for i in range(len(l1))]
            lngs_differ.append(l_differ)
        lats = lats_differ
        lngs = lngs_differ

        # cut to fit batch_size
        self.lats = lats[:len(lats) - (len(lats) % batch_size)]
        self.lngs = lngs[:len(lngs) - (len(lngs) % batch_size)]
        self.labels = labels[:len(labels) - (len(labels) % batch_size)]
        # set ptr
        self.current_ptr = 0
        self.is_done = False
        self.batch_num = len(self.lats) // batch_size
        self.batch_size = batch_size

    def batch(self):
        if not self.is_done:
            return self.get_batch()
        else:
            self.current_ptr = 0
            self.is_done = False
            return self.get_batch()

    def get_batch(self):
        lats = self.lats[self.current_ptr: self.current_ptr + self.batch_size]
        lngs = self.lngs[self.current_ptr: self.current_ptr + self.batch_size]
        features = [
            np.stack([lats[i], lngs[i]], axis=1) for i in range(len(lats))
        ]
        labels = self.labels[self.current_ptr: self.current_ptr + self.batch_size]
        features = [tf.convert_to_tensor(x, dtype=tf.float32) for x in features]
        features_tf = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        for idx in range(len(features)):
            features_tf = features_tf.write(idx, features[idx])
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return_data = features_tf, labels
        self.current_ptr += self.batch_size
        if self.current_ptr >= len(self.lats):
            self.is_done = True
        return return_data


class MyDifferDatasetWithEmbedding:
    def __init__(self, dataset, batch_size):
        lats = dataset["lats"]
        lngs = dataset["lngs"]
        labels = dataset["labels"]
        timeIDs = dataset["timeIDs"]
        weekIDs = dataset["weekIDs"]
        lats_differ = []
        lngs_differ = []
        # calculate differs
        for lat in lats:
            l1 = lat[:-1]
            l2 = lat[1:]
            l_differ = [abs(l1[i] - l2[i]) for i in range(len(l1))]
            lats_differ.append(l_differ)
        for lng in lngs:
            l1 = lng[:-1]
            l2 = lng[1:]
            l_differ = [abs(l1[i] - l2[i]) for i in range(len(l1))]
            lngs_differ.append(l_differ)
        lats = lats_differ
        lngs = lngs_differ
        timeIDs = [x[1:] for x in timeIDs]
        weekIDs = [x[1:] for x in weekIDs]
        # cut to fit batch_size
        self.lats = lats[:len(lats) - (len(lats) % batch_size)]
        self.lngs = lngs[:len(lngs) - (len(lngs) % batch_size)]
        self.labels = labels[:len(labels) - (len(labels) % batch_size)]
        self.timeIDs = timeIDs[:len(timeIDs) - (len(timeIDs) % batch_size)]
        self.weekIDs = weekIDs[:len(weekIDs) - (len(weekIDs) % batch_size)]
        # set ptr
        self.current_ptr = 0
        self.is_done = False
        self.batch_num = len(self.lats) // batch_size
        self.batch_size = batch_size

    def batch(self):
        if not self.is_done:
            return self.get_batch()
        else:
            self.current_ptr = 0
            self.is_done = False
            return self.get_batch()

    def get_batch(self):
        lats = self.lats[self.current_ptr: self.current_ptr + self.batch_size]
        lngs = self.lngs[self.current_ptr: self.current_ptr + self.batch_size]
        timeIDs = self.timeIDs[self.current_ptr: self.current_ptr + self.batch_size]
        weekIDs = self.weekIDs[self.current_ptr: self.current_ptr + self.batch_size]
        features = [
            np.stack([lats[i], lngs[i], timeIDs[i], weekIDs[i]], axis=1) for i in range(len(lats))
        ]
        labels = self.labels[self.current_ptr: self.current_ptr + self.batch_size]
        features = [tf.convert_to_tensor(x, dtype=tf.float32) for x in features]
        features_tf = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        for idx in range(len(features)):
            features_tf = features_tf.write(idx, features[idx])
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return_data = features_tf, labels
        self.current_ptr += self.batch_size
        if self.current_ptr >= len(self.lats):
            self.is_done = True
        return return_data