# Hierarchical Neural Networks (HNN) for Natural Language Processing (NLP)

## Requirements
-- Python 3.6 or higher
-- Tensorflow 1.6 or higher

## Instructions for Use
See scripts for example usage. Each model requires (1) a 2D word embedding matrix (word count x embedding size) in which the first row is set to 0 and (2) train/val/test data, each as a 2D or 3D 0-padded matrix (document x words or document x lines x words) in which each row corresponds to the word indices for a document in the train/val/test set. This repo does not include code for preprocessing data into appropriate format.
