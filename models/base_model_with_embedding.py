# -*- coding: utf-8 -*-
# @Author  : morningstarwang
# @FileName: base_model.py
# @Blog: wangchenxing.com
import tensorflow as tf

import my_config


class BaseModelWithEmbedding(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_embedding = tf.keras.layers.Embedding(24, 128)
        self.week_embedding = tf.keras.layers.Embedding(7, 128)
        self.spatial_lstm = tf.keras.layers.LSTM(128)
        self.hour_temporal_lstm = tf.keras.layers.LSTM(128)
        self.week_temporal_lstm = tf.keras.layers.LSTM(128)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            time_embeddings = self.time_embedding(tf.reshape(each_input[:, 2:3], (tf.shape(each_input[:, 2:3])[0])))
            week_embeddings = self.week_embedding(tf.reshape(each_input[:, 3:4], (tf.shape(each_input[:, 3:4])[0])))
            spatial_data = tf.expand_dims(each_input[:, :2], axis=0)
            hour_temporal_data = tf.expand_dims(time_embeddings, axis=0)
            week_temporal_data = tf.expand_dims(week_embeddings, axis=0)
            spatial_features = self.spatial_lstm(spatial_data)
            hour_temporal_features = self.hour_temporal_lstm(hour_temporal_data)
            week_temporal_features = self.week_temporal_lstm(week_temporal_data)
            x = tf.concat([spatial_features, hour_temporal_features, week_temporal_features], axis=1)
            # x = self.lstm(x)
            x = self.fc(x)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output


class Base64ModelWithEmbedding(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_embedding = tf.keras.layers.Embedding(24, 64)
        self.week_embedding = tf.keras.layers.Embedding(7, 64)
        self.spatial_lstm = tf.keras.layers.LSTM(64)
        self.hour_temporal_lstm = tf.keras.layers.LSTM(64)
        self.week_temporal_lstm = tf.keras.layers.LSTM(64)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            time_embeddings = self.time_embedding(tf.reshape(each_input[:, 2:3], (tf.shape(each_input[:, 2:3])[0])))
            week_embeddings = self.week_embedding(tf.reshape(each_input[:, 3:4], (tf.shape(each_input[:, 3:4])[0])))
            spatial_data = tf.expand_dims(each_input[:, :2], axis=0)
            hour_temporal_data = tf.expand_dims(time_embeddings, axis=0)
            week_temporal_data = tf.expand_dims(week_embeddings, axis=0)
            spatial_features = self.spatial_lstm(spatial_data)
            hour_temporal_features = self.hour_temporal_lstm(hour_temporal_data)
            week_temporal_features = self.week_temporal_lstm(week_temporal_data)
            x = tf.concat([spatial_features, hour_temporal_features, week_temporal_features], axis=1)
            # x = self.lstm(x)
            x = self.fc(x)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output


class Base256ModelWithEmbedding(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_embedding = tf.keras.layers.Embedding(24, 256)
        self.week_embedding = tf.keras.layers.Embedding(7, 256)
        self.spatial_lstm = tf.keras.layers.LSTM(256)
        self.hour_temporal_lstm = tf.keras.layers.LSTM(256)
        self.week_temporal_lstm = tf.keras.layers.LSTM(256)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        # for each_input in inputs:
        for idx in tf.range(int(my_config.general_config["batch_size"])):
            each_input = inputs.read(idx)
            time_embeddings = self.time_embedding(tf.reshape(each_input[:, 2:3], (tf.shape(each_input[:, 2:3])[0])))
            week_embeddings = self.week_embedding(tf.reshape(each_input[:, 3:4], (tf.shape(each_input[:, 3:4])[0])))
            spatial_data = tf.expand_dims(each_input[:, :2], axis=0)
            hour_temporal_data = tf.expand_dims(time_embeddings, axis=0)
            week_temporal_data = tf.expand_dims(week_embeddings, axis=0)
            spatial_features = self.spatial_lstm(spatial_data)
            hour_temporal_features = self.hour_temporal_lstm(hour_temporal_data)
            week_temporal_features = self.week_temporal_lstm(week_temporal_data)
            x = tf.concat([spatial_features, hour_temporal_features, week_temporal_features], axis=1)
            # x = self.lstm(x)
            x = self.fc(x)
            outputs.append(x)
        output = tf.stack(outputs, axis=0)
        output = tf.reshape(output, shape=[int(my_config.general_config["batch_size"]), ])
        return output
