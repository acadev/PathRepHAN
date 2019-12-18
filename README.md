# Hierarchical Neural Networks (HNN) for Natural Language Processing (NLP)

## Requirements
- Python 3.6 or higher
- Tensorflow 1.6 or higher

### Instructions to Run Models

Download the Yelp Reviews dataset from [https://www.yelp.com/dataset](https://www.yelp.com/dataset)

Use the following command to preprocess the Yelp reviews:
 - python feature_extraction_yelp.py \<path to Yelp json file\>

Use one or more of the following commands to run the desired model:
 - python tf_cnn.py
 - python tf_han.py
 - python tf_hcan.py
 - python traditional_ml.py
