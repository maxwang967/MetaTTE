# MetaTTE: a Meta-Learning Based Travel Time Estimation Model for Multi-city Scenarios

This is the official TensorFlow implementation of MetaTTE in the manuscript.

## Core Requirements
- tensorflow~=2.3.0
- numpy~=1.18.4
- spektral~=0.6.1
- pandas~=1.0.3
- tqdm~=4.46.0
- opencv-python~=4.3.0.36
- matplotlib~=3.2.1
- Pillow~=7.1.2
- scipy~=1.4.1

All Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
We here provide the datasets we adopted in this paper with Google Drive. After downloading the zip file, please extract all the files in data directory to the data folder in this project.

Download Link: <a href="https://drive.google.com/file/d/1KiiSnx5x6f8B-pkkZEk7QYHIHg7I-zp8/view?usp=sharing">Download</a>

## Configuration

We here list a sample of our config file, and leave the comments for explanation. \ 
(Please DO NOT include the comments in config files)

```
[General]
mode = train
# Specify the absoulute path of training, validation and testing files
train_files = ./data/chengdu/train.npy,./data/porto/train.npy
val_files = ./data/chengdu/val.npy,./data/porto/val.npy
test_files = ./data/chengdu/test.npy,./data/porto/test.npy
# Specify the batch size
batch_size = 32
# Specify the number for GPU
gpu = 7
# Specify the unique label for each experiment
prefix = tte_exp_64_gru

[Model]
# Specify the inner learning rate
learning_rate = 1e-2
# Specify the inner reduce rate of learning rate
lr_reduce = 0.5
# Specify the maximum iteration
epoch = 500000
# Specify the k shot
inner_k = 10
# Specify the outer step size
outer_step_size = 0.1
# Specify the model according to the class name
model = MSMTTEGRUAttModel
# Specify the dataset according to the class name
dataset = MyDifferDatasetWithEmbedding
# Specify the dataloader according to the class name
dataloader = MyDataLoaderWithEmbedding


# mean, standard deviation for latitudes, longitudes and travel time (Chengdu is before the comma while Porto is after the comma)
[Statistics]
lat_means = 30.651168872309235,41.16060653954797
lng_means = 104.06000501543934,-8.61946359614912
lat_stds = 0.039222931811691585,0.02315827641949562
lng_stds = 0.045337940910596744,0.029208656457667292
labels_means = 1088.0075248390972,691.2889878452086
labels_stds = 1315.707363003298,347.4765869900725

```

## Model Training

Here are commands for training the model on both Chengdu and Porto tasks. 

```bash
python main.py --config=./experiments/finetuning/64/gru.conf
```

## Eval baseline methods
Here are commands for testing the model on both Chengdu and Porto tasks. 
```bash
python main.py --config=./experiments/finetuning/64/gru.conf
```
## Citation

We currently do not provide citations.
