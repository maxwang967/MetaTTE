# -*- coding: utf-8 -*-
# @Author  : morningstarwang
# @FileName: dataloaders.py
# @Blog: wangchenxing.com
import numpy as np

import my_config
import sys
import datasets


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self):
        self.means = {
            "lat": list(map(float, my_config.statistics_config["lat_means"].split(","))),
            "lng": list(map(float, my_config.statistics_config["lng_means"].split(","))),
            "label": list(map(float, my_config.statistics_config["labels_means"].split(",")))
        }
        self.stds = {
            "lat": list(map(float, my_config.statistics_config["lat_stds"].split(","))),
            "lng": list(map(float, my_config.statistics_config["lng_stds"].split(","))),
            "label": list(map(float, my_config.statistics_config["labels_stds"].split(",")))
        }

    def transform(self, data, dataset_idx, data_type):
        if isinstance(data[0], list):
            return [(np.array(d) - self.means[data_type][dataset_idx]) / self.stds[data_type][dataset_idx] for d in
                    data]
        else:
            return (data - self.means[data_type][dataset_idx]) / self.stds[data_type][dataset_idx]

    # @tf.function(experimental_relax_shapes=True)
    def inverse_transform(self, data, dataset_idx, data_type):
        return (data * self.stds[data_type][dataset_idx]) + self.means[data_type][dataset_idx]


class MyDataLoader:
    def __init__(self, args):
        print("Loading data...")
        datasets = {
            "train": {},
            "val": {},
            "test": {}
        }
        self.args = args
        self.scaler = StandardScaler()
        train_files = args.general_config["train_files"].split(",")
        val_files = args.general_config["val_files"].split(",")
        test_files = args.general_config["test_files"].split(",")
        # load regular files
        self.load_regular_files(datasets, test_files, train_files, val_files)
        self.train_lens = [len(datasets["train"][k]["labels"]) for k in datasets["train"].keys()]
        self.val_lens = [len(datasets["val"][k]["labels"]) for k in datasets["val"].keys()]
        self.test_lens = [len(datasets["test"][k]["labels"]) for k in datasets["test"].keys()]
        # make our dataset
        chengdu_train_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(
            datasets["train"][0], int(my_config.general_config["batch_size"]))
        chengdu_val_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][0],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
        chengdu_test_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][0],
                                                                                                   int(
                                                                                                       my_config.general_config[
                                                                                                           "batch_size"]))
        porto_train_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["train"][0],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
        porto_val_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][0], int(
            my_config.general_config["batch_size"]))
        porto_test_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][0],
                                                                                                 int(
                                                                                                     my_config.general_config[
                                                                                                         "batch_size"]))
        # make tf datasets
        # chengdu_test_dataset, chengdu_train_dataset, chengdu_val_dataset, porto_test_dataset, porto_train_dataset, porto_val_dataset = self.make_tf_datasets()
        # Add to list
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.train_datasets.append(chengdu_train_dataset)
        self.train_datasets.append(porto_train_dataset)
        self.val_datasets.append(chengdu_val_dataset)
        self.val_datasets.append(porto_val_dataset)
        self.test_datasets.append(chengdu_test_dataset)
        self.test_datasets.append(porto_test_dataset)
        # shuffle and make batches
        self.train_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.train_datasets)
            )
        )
        self.val_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.val_datasets)
            )
        )
        self.test_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.test_datasets)
            )
        )
        print("Loading data finished.")

    def load_regular_files(self, datasets, test_files, train_files, val_files):
        # filter those uncommon range data
        filter_range = [
            (14, 141), (18, 81)
        ]
        for idx in range(len(train_files)):
            datasets["train"][idx] = {}
            train_lats = np.load(f"{train_files[idx]}-lats.npy", allow_pickle=True)
            train_lngs = np.load(f"{train_files[idx]}-lngs.npy", allow_pickle=True)
            train_labels = np.load(f"{train_files[idx]}-labels.npy", allow_pickle=True)
            train_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in train_lats]
            train_lats = train_lats[train_length_mask]
            train_lngs = train_lngs[train_length_mask]
            train_labels = train_labels[train_length_mask]
            datasets["train"][idx]["lats"] = self.scaler.transform(
                train_lats
                , idx,
                "lat")
            datasets["train"][idx]["lngs"] = self.scaler.transform(
                train_lngs
                , idx,
                "lng")
            datasets["train"][idx]["labels"] = self.scaler.transform(
                train_labels
                , idx, "label")
            datasets["val"][idx] = {}
            val_lats = np.load(f"{val_files[idx]}-lats.npy", allow_pickle=True)
            val_lngs = np.load(f"{val_files[idx]}-lngs.npy", allow_pickle=True)
            val_labels = np.load(f"{val_files[idx]}-labels.npy", allow_pickle=True)
            val_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in val_lats]
            val_lats = val_lats[val_length_mask]
            val_lngs = val_lngs[val_length_mask]
            val_labels = val_labels[val_length_mask]
            datasets["val"][idx]["lats"] = self.scaler.transform(
                val_lats, idx, "lat")
            datasets["val"][idx]["lngs"] = self.scaler.transform(
                val_lngs, idx, "lng")
            datasets["val"][idx]["labels"] = self.scaler.transform(
                val_labels, idx,
                "label")
            datasets["test"][idx] = {}
            test_lats = np.load(f"{test_files[idx]}-lats.npy", allow_pickle=True)
            test_lngs = np.load(f"{test_files[idx]}-lngs.npy", allow_pickle=True)
            test_labels = np.load(f"{test_files[idx]}-labels.npy", allow_pickle=True)
            test_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in test_lats]
            test_lats = test_lats[test_length_mask]
            test_lngs = test_lngs[test_length_mask]
            test_labels = test_labels[test_length_mask]
            datasets["test"][idx]["lats"] = self.scaler.transform(
                test_lats, idx,
                "lat")
            datasets["test"][idx]["lngs"] = self.scaler.transform(
                test_lngs, idx,
                "lng")
            datasets["test"][idx]["labels"] = self.scaler.transform(
                test_labels, idx,
                "label")


class MyAdvancedDataLoader:
    def __init__(self, args):
        print("Loading data...")
        datasets = {
            "train": {},
            "val": {},
            "test": {}
        }
        self.args = args
        self.scaler = StandardScaler()
        train_files = args.general_config["train_files"].split(",")
        val_files = args.general_config["val_files"].split(",")
        test_files = args.general_config["test_files"].split(",")
        # load regular files
        self.load_regular_files(datasets, test_files, train_files, val_files)
        self.train_lens = [len(datasets["train"][k]["labels"]) for k in datasets["train"].keys()]
        self.val_lens = [len(datasets["val"][k]["labels"]) for k in datasets["val"].keys()]
        self.test_lens = [len(datasets["test"][k]["labels"]) for k in datasets["test"].keys()]
        # make our dataset
        chengdu_train_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(
            datasets["train"][0], int(my_config.general_config["batch_size"]))
        chengdu_val_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][0],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
        chengdu_test_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][0],
                                                                                                   int(
                                                                                                       my_config.general_config[
                                                                                                           "batch_size"]))
        porto_train_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["train"][0],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
        porto_val_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][0], int(
            my_config.general_config["batch_size"]))
        porto_test_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][0],
                                                                                                 int(
                                                                                                     my_config.general_config[
                                                                                                         "batch_size"]))
        # make tf datasets
        # chengdu_test_dataset, chengdu_train_dataset, chengdu_val_dataset, porto_test_dataset, porto_train_dataset, porto_val_dataset = self.make_tf_datasets()
        # Add to list
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.train_datasets.append(chengdu_train_dataset)
        self.train_datasets.append(porto_train_dataset)
        self.val_datasets.append(chengdu_val_dataset)
        self.val_datasets.append(porto_val_dataset)
        self.test_datasets.append(chengdu_test_dataset)
        self.test_datasets.append(porto_test_dataset)
        # shuffle and make batches
        self.train_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.train_datasets)
            )
        )
        self.val_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.val_datasets)
            )
        )
        self.test_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.test_datasets)
            )
        )
        print("Loading data finished.")

    def load_regular_files(self, datasets, test_files, train_files, val_files):
        # filter those uncommon range data
        filter_range = [
            (14, 141), (18, 81)
        ]
        for idx in range(len(train_files)):
            datasets["train"][idx] = {}
            train_lats = np.load(f"{train_files[idx]}-lats.npy", allow_pickle=True)
            train_lngs = np.load(f"{train_files[idx]}-lngs.npy", allow_pickle=True)
            train_dis = np.load(f"{train_files[idx]}-dis.npy", allow_pickle=True)
            train_labels = np.load(f"{train_files[idx]}-labels.npy", allow_pickle=True)
            train_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in train_lats]
            train_lats = train_lats[train_length_mask]
            train_lngs = train_lngs[train_length_mask]
            train_dis = train_dis[train_length_mask]
            train_labels = train_labels[train_length_mask]
            datasets["train"][idx]["lats"] = self.scaler.transform(
                train_lats
                , idx,
                "lat")
            datasets["train"][idx]["lngs"] = self.scaler.transform(
                train_lngs
                , idx,
                "lng")
            datasets["train"][idx]["dis"] = self.scaler.transform(
                train_dis
                , idx,
                "dis")
            datasets["train"][idx]["labels"] = self.scaler.transform(
                train_labels
                , idx, "label")
            datasets["val"][idx] = {}
            val_lats = np.load(f"{val_files[idx]}-lats.npy", allow_pickle=True)
            val_lngs = np.load(f"{val_files[idx]}-lngs.npy", allow_pickle=True)
            val_dis = np.load(f"{val_files[idx]}-dis.npy", allow_pickle=True)
            val_labels = np.load(f"{val_files[idx]}-labels.npy", allow_pickle=True)
            val_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in val_lats]
            val_lats = val_lats[val_length_mask]
            val_lngs = val_lngs[val_length_mask]
            val_dis = val_dis[val_length_mask]
            val_labels = val_labels[val_length_mask]
            datasets["val"][idx]["lats"] = self.scaler.transform(
                val_lats, idx, "lat")
            datasets["val"][idx]["lngs"] = self.scaler.transform(
                val_lngs, idx, "lng")
            datasets["val"][idx]["dis"] = self.scaler.transform(
                val_dis, idx, "dis")
            datasets["val"][idx]["labels"] = self.scaler.transform(
                val_labels, idx,
                "label")
            datasets["test"][idx] = {}
            test_lats = np.load(f"{test_files[idx]}-lats.npy", allow_pickle=True)
            test_lngs = np.load(f"{test_files[idx]}-lngs.npy", allow_pickle=True)
            test_dis = np.load(f"{test_files[idx]}-dis.npy", allow_pickle=True)
            test_labels = np.load(f"{test_files[idx]}-labels.npy", allow_pickle=True)
            test_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in test_lats]
            test_lats = test_lats[test_length_mask]
            test_lngs = test_lngs[test_length_mask]
            test_dis = test_dis[test_length_mask]
            test_labels = test_labels[test_length_mask]
            datasets["test"][idx]["lats"] = self.scaler.transform(
                test_lats, idx,
                "lat")
            datasets["test"][idx]["lngs"] = self.scaler.transform(
                test_lngs, idx,
                "lng")
            datasets["test"][idx]["dis"] = self.scaler.transform(
                test_dis, idx,
                "dis")
            datasets["test"][idx]["labels"] = self.scaler.transform(
                test_labels, idx,
                "label")


class MyDataLoaderWithEmbedding:
    def __init__(self, args):
        print("Loading data...")
        datasets = {
            "train": {},
            "val": {},
            "test": {}
        }
        self.args = args
        self.scaler = StandardScaler()
        train_files = args.general_config["train_files"].split(",")
        val_files = args.general_config["val_files"].split(",")
        test_files = args.general_config["test_files"].split(",")
        # load regular files
        self.load_regular_files(datasets, test_files, train_files, val_files)
        self.train_lens = [len(datasets["train"][k]["labels"]) for k in datasets["train"].keys()]
        self.val_lens = [len(datasets["val"][k]["labels"]) for k in datasets["val"].keys()]
        self.test_lens = [len(datasets["test"][k]["labels"]) for k in datasets["test"].keys()]
        # make our dataset
        chengdu_train_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(
            datasets["train"][0], int(my_config.general_config["batch_size"]))
        chengdu_val_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][0],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
        chengdu_test_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][0],
                                                                                                   int(
                                                                                                       my_config.general_config[
                                                                                                           "batch_size"]))
        porto_train_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["train"][0],
                                                                                                  int(
                                                                                                      my_config.general_config[
                                                                                                          "batch_size"]))
        porto_val_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["val"][0], int(
            my_config.general_config["batch_size"]))
        porto_test_dataset = getattr(sys.modules["datasets"], my_config.model_config["dataset"])(datasets["test"][0],
                                                                                                 int(
                                                                                                     my_config.general_config[
                                                                                                         "batch_size"]))
        # make tf datasets
        # chengdu_test_dataset, chengdu_train_dataset, chengdu_val_dataset, porto_test_dataset, porto_train_dataset, porto_val_dataset = self.make_tf_datasets()
        # Add to list
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.train_datasets.append(chengdu_train_dataset)
        self.train_datasets.append(porto_train_dataset)
        self.val_datasets.append(chengdu_val_dataset)
        self.val_datasets.append(porto_val_dataset)
        self.test_datasets.append(chengdu_test_dataset)
        self.test_datasets.append(porto_test_dataset)
        # shuffle and make batches
        self.train_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.train_datasets)
            )
        )
        self.val_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.val_datasets)
            )
        )
        self.test_datasets = list(
            map(
                lambda x: (
                    x[0],
                    x[1]),
                enumerate(self.test_datasets)
            )
        )
        print("Loading data finished.")

    def load_regular_files(self, datasets, test_files, train_files, val_files):
        # filter those uncommon range data
        filter_range = [
            (14, 141), (18, 81)
        ]
        for idx in range(len(train_files)):
            datasets["train"][idx] = {}
            train_lats = np.load(f"{train_files[idx]}-lats.npy", allow_pickle=True)
            train_lngs = np.load(f"{train_files[idx]}-lngs.npy", allow_pickle=True)
            train_timeIDs = np.load(f"{train_files[idx]}-timeID.npy", allow_pickle=True)
            train_weekIDs = np.load(f"{train_files[idx]}-weekID.npy", allow_pickle=True)
            train_labels = np.load(f"{train_files[idx]}-labels.npy", allow_pickle=True)
            train_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in train_lats]
            train_lats = train_lats[train_length_mask]
            train_lngs = train_lngs[train_length_mask]
            train_timeIDs = train_timeIDs[train_length_mask]
            train_weekIDs = train_weekIDs[train_length_mask]
            train_labels = train_labels[train_length_mask]
            datasets["train"][idx]["lats"] = self.scaler.transform(
                train_lats
                , idx,
                "lat")
            datasets["train"][idx]["lngs"] = self.scaler.transform(
                train_lngs
                , idx,
                "lng")
            datasets["train"][idx]["timeIDs"] = train_timeIDs
            datasets["train"][idx]["weekIDs"] = train_weekIDs
            datasets["train"][idx]["labels"] = self.scaler.transform(
                train_labels
                , idx, "label")
            datasets["val"][idx] = {}
            val_lats = np.load(f"{val_files[idx]}-lats.npy", allow_pickle=True)
            val_lngs = np.load(f"{val_files[idx]}-lngs.npy", allow_pickle=True)
            val_labels = np.load(f"{val_files[idx]}-labels.npy", allow_pickle=True)
            val_timeIDs = np.load(f"{val_files[idx]}-timeID.npy", allow_pickle=True)
            val_weekIDs = np.load(f"{val_files[idx]}-weekID.npy", allow_pickle=True)
            val_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in val_lats]
            val_lats = val_lats[val_length_mask]
            val_lngs = val_lngs[val_length_mask]
            val_timeIDs = val_timeIDs[val_length_mask]
            val_weekIDs = val_weekIDs[val_length_mask]
            val_labels = val_labels[val_length_mask]
            datasets["val"][idx]["lats"] = self.scaler.transform(
                val_lats, idx, "lat")
            datasets["val"][idx]["lngs"] = self.scaler.transform(
                val_lngs, idx, "lng")
            datasets["val"][idx]["labels"] = self.scaler.transform(
                val_labels, idx,
                "label")
            datasets["val"][idx]["timeIDs"] = val_timeIDs
            datasets["val"][idx]["weekIDs"] = val_weekIDs
            datasets["test"][idx] = {}
            test_lats = np.load(f"{test_files[idx]}-lats.npy", allow_pickle=True)
            test_lngs = np.load(f"{test_files[idx]}-lngs.npy", allow_pickle=True)
            test_labels = np.load(f"{test_files[idx]}-labels.npy", allow_pickle=True)
            test_timeIDs = np.load(f"{test_files[idx]}-timeID.npy", allow_pickle=True)
            test_weekIDs = np.load(f"{test_files[idx]}-weekID.npy", allow_pickle=True)
            test_length_mask = [filter_range[idx][1] >= len(x) >= filter_range[idx][0] for x in test_lats]
            test_lats = test_lats[test_length_mask]
            test_lngs = test_lngs[test_length_mask]
            test_timeIDs = test_timeIDs[test_length_mask]
            test_weekIDs = test_weekIDs[test_length_mask]
            test_labels = test_labels[test_length_mask]
            datasets["test"][idx]["lats"] = self.scaler.transform(
                test_lats, idx,
                "lat")
            datasets["test"][idx]["lngs"] = self.scaler.transform(
                test_lngs, idx,
                "lng")
            datasets["test"][idx]["timeIDs"] = test_timeIDs
            datasets["test"][idx]["weekIDs"] = test_weekIDs
            datasets["test"][idx]["labels"] = self.scaler.transform(
                test_labels, idx,
                "label")

