# ResNet-LSTM-model-for-Violence-Detection

## Introduction

This repo is the implemetation of a research paper for Violence detection using deep learning techniques. The said paper is available on this repo. The dataset used for training
this model was made by the authors of the paper. The videos choosen were from the context of south-east asian violence and was collected from various sites like Facebook, Youtube
as well as news.

## Procedure

The entire process of the repo or the paper is of 4 steps.

* Data preprocessing
* Extraction of features by a pre-trained model
* Training a model using the Features
* Testing the model's accuracy.

### Data preprocessing

Frames were extracted from the videos. These frames were the initial dataset which would be used for training and testing the model. The extracted frames were prelabeled
according to the videos where the frames were extracted from. The dataset frames were imbalanced so we down-sampled the extra samples and trained the model with a balanced 
dataset. 

### Extraction of features by a pre-trained model

The pre-trained model used was ResNet. As ResNet is a very good model for object detection in image, we used this to extract key features from each frames. We imported the ResNet
model from keras and instantiated the model without the top. The extracted features were bundle together to be send on to the next layer of our model which is the LSTM layer.

### Training the model using the features

The feature information were bundled together in a sequential manner as these need to be send to an LSTM layer. The sequence length was 30. The LSTM used by the authors of the
paper was CuDNNLSTM, i.e. using the GPU to fasten up the time taken to train. The output of the LSTM layer were fed into a fully connected layer with 1 neuron only. This would 
determine the binary classificaion of our model. 1 being non-violent and 0 being the video is violent.

### Testing the model's accuracy

The model was validated with a validation split of 0.1 where 90% of the frames were used for training the model. The model were later tested using 2 other bencemark
violence detection datasets, the hockey fights and the movie fights datasets. This was to see the model's accuracy with another type of videos.

## Results

The result of our model was a bit better than the authors being of 98.7% validation accuracy of the dataset. 
