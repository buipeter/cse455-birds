---
layout: default
title: CSE 455 Final Project
---

# The Problem

The objective of this project is to develop a machine learning model capable of accurately identifying the species of birds by analyzing the characteristics present in an image. By accomplishing this, the aim is to create a sense of familiarity with these magnificent flying creatures, which have undergone diverse adaptations over millions of years, paralleling the evolution of humans and becoming one of the most varied animal groups on Earth, encompassing over 5,000 distinct species.

# Introduction

In our project, we initially utilized Google Colab as our working environment for a certain duration before transitioning to Jupyter Notebook. We selected the ResNet18 model as our initial pre-trained framework and later expanded our exploration by incorporating DenseNet161. For the actual training process, computation of loss values, and experimentation with various training techniques, we relied on the PyTorch Python library.

We decided to train and utilize a learning model to classify 10,000 images of birds from a dataset provided by the biannual [Bird Classification Kaggle Competition](https://www.kaggle.com/competitions/birds23sp/data). However, this begged the question: which model should we use to achieve a desired level of classification and how should we adjust this model to perform optimally and accurately?