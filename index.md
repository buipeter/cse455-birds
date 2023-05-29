---
layout: default
title: CSE 455 Final Project
---

# The Problem

The objective of this project is to develop a machine learning model capable of accurately identifying the species of birds by analyzing the characteristics present in an image. By accomplishing this, the aim is to create a sense of familiarity with these magnificent flying creatures, which have undergone diverse adaptations over millions of years, paralleling the evolution of humans and becoming one of the most varied animal groups on Earth, encompassing over 5,000 distinct species.

# Introduction

In our project, I initially utilized Google Colab as our working environment for a certain duration before transitioning to Jupyter Notebook. I selected the ResNet18 model as our initial pre-trained framework and later expanded our exploration by incorporating DenseNet161. For the actual training process, computation of loss values, and experimentation with various training techniques, I relied on the PyTorch Python library.

I decided to train and utilize a learning model to classify 10,000 images of birds from a dataset provided by the biannual [Bird Classification Kaggle Competition](https://www.kaggle.com/competitions/birds23sp/data). However, this begged the question: which model should I use to achieve a desired level of classification and how should I adjust this model to perform optimally and accurately?

# How It Started

I began by using the Google Colab as the environment for training my model. I used the dataset that was provided by us from the kaggle website, from then I utilized our code from the classes Pytorch Tutorial to load the dataset and resized the images resolution to 128 pixel by 128 pixel. I kept the training batch size to 128, which was the same as the Pytorch's tutorial, and then modified the model to run for 6 epochs with a learning rate of 0.01 then decreasing to 0.001 after the fifth epoch and a decay of 0.0005. This was the very inital start of the testing that I took so that I could grasp the numbers and the length of each epoch's training.