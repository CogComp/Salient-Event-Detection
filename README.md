# Salient-Event-Detection
The repository for the paper "Is Killed More Significant than Fled? A Contextual Model for Salient Event Detection"

<p align="center"><img src="Overview.png" width="800"></p>

## Abstract

Identifying the key events in a document is critical to holistically understanding its important information. Although measuring the salience of events is highly contextual, most previous work has used a limited representation of events that omits essential information. In this work, we propose a highly contextual model of event salience that uses a rich representation of events, incorporates document-level information and allows for interactions between latent event encod- ings. Our experimental results on an event salience dataset (Liu et al., 2018) demonstrate that our model improves over previous work by an absolute 2-4% on standard metrics, establishing a new state-of-the-art performance for the task. We also propose a new evaluation metric which addresses flaws in previous evaluation methodologies. Finally, we discuss the importance of salient event detection for the downstream task of summarization.

## Data
[Annotated NYT](https://catalog.ldc.upenn.edu/LDC2008T19) is the raw dataset used in this work. [Here](TODO) is the train/validation/test split.

## Code

### Requirements

### Train

1. Clone the code [repository](https://github.com/DishaJindal/Salient-Event-Detection).
1. Pass data path flags: `` train_data_path,  val_data_path, test_data_path `` with the appropriate locations while training.
1. Train using `` python CEE-train.py `` by passing desired flags. Here is a sample invocation:

  `` python CEE-train.py "BERT-SVA" model_name "0,1" -sl -nem -ps -frame ``

### Run inference

1. Clone the code [repository](https://github.com/DishaJindal/Salient-Event-Detection).
1. Pass data path flags: `` test_dir and test_file `` with the test file path and name respectively along with the feature flags same as that of trained model.
1. Run inference using `` python CEE-predict.py `` by passing desired flags. Here is a sample invocation:

  `` python CEE-predict.py "BERT-SVA" "0,1" path_to_the_saved_model output_file_path -sl -nem -ps -frame ``

## Results

This [folder](https://github.com/DishaJindal/Salient-Event-Detection/tree/master/results) contains the document wise predictions of all models on the test set. 
The predictions of Location, Frequncy and KCM baselines can be found in the [baselines](https://github.com/DishaJindal/Salient-Event-Detection/tree/master/results/baselines) folder and 
of feature ablation models in the [ablations](https://github.com/DishaJindal/Salient-Event-Detection/tree/master/results/ablations) folder.

[cee_iee.json.zip](https://github.com/DishaJindal/Salient-Event-Detection/tree/master/results/cee_iee.json.zip) contains the predictions of the model with the best performance.




