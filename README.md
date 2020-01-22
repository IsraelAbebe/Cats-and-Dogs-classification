# Cats And Dogs Image Classification in Pytorch
--------------------------------


The Dogs vs. Cats dataset is a standard computer vision dataset that involves classifying photos as either containing a dog or cat.Our basic task is to create an algorithm to classify whether an image contains a dog or a cat. The
input for this task is images of dogs or cats from training dataset, while the output is the classification
accuracy on test dataset.[[1]](https://sites.ualberta.ca/~bang3/files/DogCat_report.pdf)


## Installation
---------------------
  
    - pip
    >> pip install -r requirement.txt

    - Conda
    >> conda env create -f environment.yml
    >> conda activate cats_and_dogs
    
 
## Downloading Dataset
---------------------

    >> cd data
    >> bash download.sh

    
## Usage
  
    >> python main.py -h
       usage: main.py [-h] [-file_dir FILE_DIR] [-batch_size BATCH_SIZE] [-lr LR]
              [-epoch EPOCH] [-model {SIMPLE,DEEPER}]

    #Example

    >> python main.py -epoch 10 -model 'DEEPER'
    
   
   
## Tensorboard Visualizations
------------------------

    >> tensorboard --logdir='TensorBoard/runs/'

### Images
----------------

<img width="600" alt="Images" src="data/images/images.png">


### Architecture
----------------

<img width="600" alt="Architecture" src="data/images/cnn.png">



### Train and test Loss
----------------

<img width="600" alt="Test Loss" src="data/images/training_loss1.png">


<img width="600" alt="Train and test Loss" src="data/images/test_loss.png">


### Visualizations
------------------

<img width="600" alt="Visualizations" src="data/images/tsne.png" >


### Output
----------------
<img width="600" alt="output" src="data/images/vs.png" >


### Output
----------------

<img width="600" alt="output" src="data/images/pr1.png" >


<img width="600" alt="output" src="data/images/pr2.png" >




