# -*- coding: utf-8 -*-
# @Author  : morningstarwang
# @FileName: base_model.py
# @Blog: wangchenxing.com
import tensorflow as tf

import my_config


class BaseModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = tf.keras.layers.LSTM(128)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            x = tf.expand_dims(each_input, axis=0)
            x = self.lstm(x)
            x = self.fc(x)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output


class Base64Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = tf.keras.layers.LSTM(64)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            x = tf.expand_dims(each_input, axis=0)
            x = self.lstm(x)
            x = self.fc(x)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output


class Base256Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = tf.keras.layers.LSTM(256)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            x = tf.expand_dims(each_input, axis=0)
            x = self.lstm(x)
            x = self.fc(x)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output