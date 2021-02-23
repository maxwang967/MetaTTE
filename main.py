# -*- coding: utf-8 -*-
# @Author  : morningstarwang
# @FileName: main.py
# @Blog: wangchenxing.com

import configparser
import argparse
import datetime
import os
import sys
import random
import numpy as np
import models

import tensorflow as tf
from tensorflow.python.keras import backend as K

import my_config
from dataloaders import MyDataLoader
from base_utils import lr_fn, masked_mae_tf, masked_mape_tf, masked_rmse_tf
from tensorflow.keras.utils import Progbar

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="configuration file path", required=False, default="./config/default.conf")
parser.add_argument("--test_file", type=str,
                    help="specify test file in the highest priority", required=False, default="None")
args = parser.parse_args()
# read configuration
config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
config.read(args.config)
test_file_str = args.test_file
my_config.general_config = config['General']
my_config.model_config = config['Model']
my_config.statistics_config = config['Statistics']
gpu = int(my_config.general_config["gpu"])
mode = my_config.general_config["mode"]
tf.keras.backend.set_floatx('float32')
gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices(devices=[gpus[gpu], cpus[0]])

def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        output = our_model(x)
        loss = masked_mae_tf(y, output)
        grads = tape.gradient(loss, our_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, our_model.trainable_variables))
    return loss, output, y

def test_on_batch(x, y):
    output = our_model(x)
    return output

def train():
    my_config.is_training = True
    chengdu_initial_loss = 1e9
    porto_initial_loss = 1e9
    print("Loading Weights from checkpoints...")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("Loading Finished.")
    test_epoch = 0
    for epoch in range(epochs):
        lr = optimizer.learning_rate.numpy()
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        weights_before = our_model.get_weights()
        task_dataset = random.sample(dataloader.train_datasets, 1)[0]
        pb_i = Progbar(inner_k, stateful_metrics=metrics_names)
        for k in range(inner_k):
            features, labels = task_dataset[1].batch()
            output, y = None, None
            if task_dataset[0] == 0:
                # loss, output, y = train_on_batch(features)
                loss, output, y = train_on_batch(features, labels)
            elif task_dataset[0] == 1:
                # loss, output, y = train_on_batch(features)
                loss, output, y = train_on_batch(features, labels)
            else:
                # TODO you may add more tasks here
                loss, mae, mape, rmse = -1
            mae = masked_mae_tf(dataloader.scaler.inverse_transform(y, task_dataset[0], "label"),
                                dataloader.scaler.inverse_transform(output, task_dataset[0], "label"))
            mape = masked_mape_tf(dataloader.scaler.inverse_transform(y, task_dataset[0], "label"),
                                  dataloader.scaler.inverse_transform(output, task_dataset[0], "label"))
            rmse = masked_rmse_tf(dataloader.scaler.inverse_transform(y, task_dataset[0], "label"),
                                  dataloader.scaler.inverse_transform(output, task_dataset[0], "label"))
            loss_metrics.update_state(loss)
            mae_metrics.update_state(mae)
            mape_metrics.update_state(mape)
            rmse_metrics.update_state(rmse)
            pb_i.add(1, values=[
                ("LOSS", loss), ('MAE', mae), ('MAPE', mape), ('RMSE', rmse)
            ])
        epoch_loss = loss_metrics.result()
        epoch_mae = mae_metrics.result()
        epoch_mape = mape_metrics.result()
        epoch_rmse = rmse_metrics.result()
        loss_metrics.reset_states()
        mae_metrics.reset_states()
        mape_metrics.reset_states()
        rmse_metrics.reset_states()
        weights_after = our_model.get_weights()
        outer_step_size_calcu = outer_step_size * (1 - epoch / epochs)
        our_model.set_weights(
            [weights_before[i] + (weights_after[i] - weights_before[i]) * outer_step_size_calcu
             for i in range(len(our_model.weights))]
        )
        with train_summary_writer.as_default():
            tf.summary.scalar(f'{"Chengdu" if task_dataset[0] == 0 else "Porto"} Loss', epoch_loss.numpy(),
                              step=epoch)
            tf.summary.scalar(f'{"Chengdu" if task_dataset[0] == 0 else "Porto"} MAE', epoch_mae.numpy(),
                              step=epoch)
        print(f'Task {"Chengdu" if task_dataset[0] == 0 else "Porto"}:')
        print(
            f"EPOCH_MAE:{epoch_mae}, EPOCH_MAPE:{epoch_mape}, EPOCH_RMSE:{epoch_rmse}")

        if (epoch + 1) % 1000 == 0:
            test_epoch += 1
            changed_lr = lr_fn(epoch, lr_reduce, lr)
            print('changed_lr:', changed_lr)
            K.set_value(optimizer.lr, changed_lr)
            print('validation begin:')
            chengdu_loss, porto_loss = test(is_testing=False, epoch=test_epoch)
            print('validation end.')
            if chengdu_loss < chengdu_initial_loss:
                checkpoint.save(file_prefix=checkpoint_prefix_chengdu)
                chengdu_initial_loss = chengdu_loss
            if porto_loss < porto_initial_loss:
                checkpoint.save(file_prefix=checkpoint_prefix_porto)
                porto_initial_loss = porto_loss

def test(is_testing=True, epoch=0):
    if test_file_str != "None":
        print("Testing mode on: " + test_file_str)
        test_filename = test_file_str.replace('\r', '')
        test_log_file = open(f"./experiments/results/{my_config.general_config['prefix']}/{test_filename}.txt", "a+")
    my_config.is_training = False
    mae_loss = 0.0
    mape_loss = 0.0
    rmse_loss = 0.0
    # TODO This part can be refactored to a more reasonable pattern if more datasets are added
    # Chengdu Part
    if is_testing:
        print("Loading Weights from chengdu checkpoints...")
        dataset = dataloader.test_datasets[0]
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir + "_chengdu"))
        print("Loading Chengdu Finished.")
    else:
        dataset = dataloader.val_datasets[0]
    print("Chengdu Begin:")
    pb_i = Progbar(dataset[1].batch_num, stateful_metrics=metrics_names)
    for _ in range(dataset[1].batch_num):
        x, y = dataset[1].batch()
        output = test_on_batch(x, y)
        mae = masked_mae_tf(
            dataloader.scaler.inverse_transform(y, dataset[0], "label"),
            dataloader.scaler.inverse_transform(output, dataset[0], "label"),
        )
        mape = masked_mape_tf(
            dataloader.scaler.inverse_transform(y, dataset[0], "label"),
            dataloader.scaler.inverse_transform(output, dataset[0], "label"))
        rmse = masked_rmse_tf(
            dataloader.scaler.inverse_transform(y, dataset[0], "label"),
            dataloader.scaler.inverse_transform(output, dataset[0], "label"))
        mae_loss += mae
        mape_loss += mape
        rmse_loss += rmse
        pb_i.add(1, values=[
            ('MAE', mae), ('MAPE', mape), ('RMSE', rmse)
        ])
    chengdu_epoch_mae = mae_loss / dataset[1].batch_num
    chengdu_epoch_mape = mape_loss / dataset[1].batch_num
    chengdu_epoch_rmse = rmse_loss / dataset[1].batch_num
    if test_file_str == "None":
        with test_summary_writer.as_default():
            tf.summary.scalar(f'Chengdu Test MAE', chengdu_epoch_mae,
                              step=epoch)
            tf.summary.scalar(f'Chengdu Test MAPE', chengdu_epoch_mape,
                              step=epoch)
            tf.summary.scalar(f'Chengdu Test RMSE', chengdu_epoch_rmse,
                              step=epoch)
    else:
        test_log_file.write(
            f"Chengdu: EPOCH_MAE: {chengdu_epoch_mae}, EPOCH_MAPE: {chengdu_epoch_mape}, EPOCH_RMSE: {chengdu_epoch_rmse}\n")
    print(f"EPOCH_MAE: {chengdu_epoch_mae}, EPOCH_MAPE: {chengdu_epoch_mape}, EPOCH_RMSE: {chengdu_epoch_rmse}")
    print("Chengdu End.")
    mae_loss = 0.0
    mape_loss = 0.0
    rmse_loss = 0.0
    # Porto Part
    if is_testing:
        print("Loading Weights from porto checkpoints...")
        dataset = dataloader.test_datasets[1]
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir + "_porto"))
        print("Loading Chengdu Finished.")
    else:
        dataset = dataloader.val_datasets[1]
    print("Porto Begin:")
    pb_i = Progbar(dataset[1].batch_num, stateful_metrics=metrics_names)
    for _ in range(dataset[1].batch_num):
        x, y = dataset[1].batch()
        output = test_on_batch(x, y)
        mae = masked_mae_tf(
            dataloader.scaler.inverse_transform(y, dataset[0], "label"),
            dataloader.scaler.inverse_transform(output, dataset[0], "label"),
        )
        mape = masked_mape_tf(
            dataloader.scaler.inverse_transform(y, dataset[0], "label"),
            dataloader.scaler.inverse_transform(output, dataset[0], "label"))
        rmse = masked_rmse_tf(
            dataloader.scaler.inverse_transform(y, dataset[0], "label"),
            dataloader.scaler.inverse_transform(output, dataset[0], "label"))
        mae_loss += mae
        mape_loss += mape
        rmse_loss += rmse
        pb_i.add(1, values=[
            ('MAE', mae), ('MAPE', mape), ('RMSE', rmse)
        ])
    porto_epoch_mae = mae_loss / dataset[1].batch_num
    porto_epoch_mape = mape_loss / dataset[1].batch_num
    porto_epoch_rmse = rmse_loss / dataset[1].batch_num
    if test_file_str == "None":
        with test_summary_writer.as_default():
            tf.summary.scalar(f'Porto Test MAE', porto_epoch_mae,
                              step=epoch)
            tf.summary.scalar(f'Porto Test MAPE', porto_epoch_mape,
                              step=epoch)
            tf.summary.scalar(f'Porto Test RMSE', porto_epoch_rmse,
                              step=epoch)
    else:
        test_log_file.write(
            f"Porto: EPOCH_MAE: {porto_epoch_mae}, EPOCH_MAPE: {porto_epoch_mape}, EPOCH_RMSE: {porto_epoch_rmse}\n")
    print(f"EPOCH_MAE: {porto_epoch_mae}, EPOCH_MAPE: {porto_epoch_mape}, EPOCH_RMSE: {porto_epoch_rmse}")
    print("Porto End.")
    if test_file_str != "None":
        print("Testing mode end")
        test_log_file.close()
    return chengdu_epoch_mae, porto_epoch_mae


if __name__ == '__main__':
    batch_size = int(my_config.general_config["batch_size"])
    learning_rate = float(my_config.model_config["learning_rate"])
    lr_reduce = float(my_config.model_config["lr_reduce"])
    epochs = int(my_config.model_config["epoch"])
    inner_k = int(my_config.model_config["inner_k"])
    outer_step_size = float(my_config.model_config["outer_step_size"])
    metrics_names = ['LOSS', 'MAE', "MAPE", "RMSE"]
    if test_file_str != "None":
        test_str_split = test_file_str.replace("\r", "").split(",")
        data_type1 = test_str_split[0]
        data_type2 = test_str_split[1]
        my_config.general_config[
            "test_files"] = f"./data/chengdu/{data_type1}/{data_type2}/test.npy,./data/porto/{data_type1}/{data_type2}/test.npy"
        my_config.general_config[
            "train_files"] = f"./data/chengdu/{data_type1}/{data_type2}/test.npy,./data/porto/{data_type1}/{data_type2}/test.npy"
        my_config.general_config[
            "val_files"] = f"./data/chengdu/{data_type1}/{data_type2}/test.npy,./data/porto/{data_type1}/{data_type2}/test.npy"
        print(f"Test files changed to: {my_config.general_config['test_files']}")
    dataloader = getattr(sys.modules["dataloaders"], my_config.model_config["dataloader"])(my_config)
    our_model = getattr(sys.modules["models"], my_config.model_config["model"])()
    # we need to initialize the weights
    dummy_input = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    our_model(dummy_input.unstack([tf.zeros(
        shape=(100, 4), dtype=tf.float32
    ) for _ in range(batch_size)])
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_metrics = tf.keras.metrics.Mean()
    mae_metrics = tf.keras.metrics.Mean()
    mape_metrics = tf.keras.metrics.Mean()
    rmse_metrics = tf.keras.metrics.Mean()
    checkpoint_dir = f'./checkpoints_{my_config.general_config["prefix"]}'
    if not os.path.exists(checkpoint_dir + "_chengdu"):
        os.mkdir(checkpoint_dir + "_chengdu")
    if not os.path.exists(checkpoint_dir + "_porto"):
        os.mkdir(checkpoint_dir + "_porto")
    checkpoint_prefix_chengdu = os.path.join(checkpoint_dir + "_chengdu", my_config.general_config["prefix"])
    checkpoint_prefix_porto = os.path.join(checkpoint_dir + "_porto", my_config.general_config["prefix"])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=our_model)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if test_file_str == "None":
        train_log_dir = 'logs/gradient_tape/' + my_config.general_config["prefix"] + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + my_config.general_config["prefix"] + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    if "train" == mode:
        train()
    elif "test" == mode:
        test()
