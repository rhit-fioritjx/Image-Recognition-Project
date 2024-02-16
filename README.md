# CSSE463Project - PictureMath

## Overview
* This repository contains all the Jupyter notebooks and 2 out of 3 datasets used for our project. The last dataset can be found [here](https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/data)
* The main files in this repo:
  * parse_equation_test.ipynb - script used to parse math expression image into individual characters
  * classifier_2_best_without_fine_tuning.ipynb - builds and trains a simple CNN using the Kaggle dataset linked above. save the best model during training as MATH.h5
  * fine_tuning.ipynb - used previous trained CNN MATH.h5 and fine-tune-dataset to fine tune the CNN to our handwritten math characters
  * demo.ipynb - used for in-class demo
  * TrOCR training
  * fine-tune-dataset - the data used for fine-tuning and testing
  * offline-crohme - the data used for fine-tuning TrORC model
    * This is a git submodule, so will need to be pulled with `git pull --recurse-submodules`
  
## Packages
* tensorflow
* sklearn
* latex2sympy2
* ipywidgets
* cv2
* pandas
* matplotlib

* TROCR is setup using poetry
  * Run `poetry install` to setup a virtual environment with the required dependencies


## Gettting Started

* ### Running [parse_equation_test.ipynb](https://github.com/rhit-fioritjx/Image-Recognition-Project/blob/main/parse_equation_test.ipynb)
  * Make sure `test.png` is on the same directory level as the notebook
  * Run the notebook

* ### Running [classifier_2_best_without_fine_tuning.ipynb](https://github.com/rhit-fioritjx/Image-Recognition-Project/blob/main/classifier_2_best_without_fine_tuning.ipynb)
  * Make sure to down the `CompleteDataSet_training_tuples.npy`, `CompleteDataSet_validation_tuples.npy`, and `CompleteDataSet_testing_tuples.npy` from Kaggle dataset and put them into a folder called data in same level as the notebook
  * Run the notebook and `MATH.h5` should be saved in current directory

* ### Runing [fine_tuning.ipynb](https://github.com/rhit-fioritjx/Image-Recognition-Project/blob/main/fine_tuning.ipynb)
  * Make sure `MATH.h5` is on the same directory level as the notebook
  * Make sure fine-tune-dataset is on the same directory level as the notebook
  * Run the notebook and `fine_tuned_model.keras` should be saved in current directory

* ### Runing [demo.ipynb](https://github.com/rhit-fioritjx/Image-Recognition-Project/blob/main/demo.ipynb)
  * Make sure `handwritten-full-test` is on the same directory level as the notebook
  * Run the notebook
  * You can try your own images by changing that image path in 6th code chunk
 
* ### Running [TrOCR_train.py](https://github.com/rhit-fioritjx/Image-Recognition-Project/blob/main/TrOCR_train.py)
  * Make sure you run using a virtual environment with all the needed dependencies
  * We reccomend running this in a `screen` session so that you do not need to keep the session running
  * Run the file and wait roughly 4 hours
  * This will save off training curves and model checkpoints to seq2seq_model_printed
  * Modify any files using the model to use your most recent checkpoint
  
