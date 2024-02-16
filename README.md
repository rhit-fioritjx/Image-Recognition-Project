# CSSE463Project - PictureMath

## Overview
* This repository contains all the Jupyter notebooks and 2 out of 3 datasets used for our project. The last dataset can be found [here](https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/data)
* The main files in this repo:
  * parse_equation_test.ipynb - script used to parse math expression image into individual characters
  * classifier_2_best_without_fine_tuning.ipynb - builds and trains a simple CNN using the Kaggle dataset linked above. save the best model during training as MATH.h5
  * fine_tuning.ipynb - used previous trained CNN MATH.h5 and fine-tune-dataset to fine tune the CNN to our handwritten math characters
  * TrOCR??
  * fine-tune-dataset - the data used for fine-tuning and testing
  * offline-crohme - the data used for fine-tuning TrORC model
  
## Packages
not too sure about TrOCR


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
  
