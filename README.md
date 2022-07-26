# Heart Disease Prediction Using FeedForward Neural Network (TensorFlow)

## 1. Summary

The aim of this project is to create a deep learning model to predict whether a patient has heart disease or not.

The model is trained with the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) that was obtained from kaggle.

## 2. IDE and Framework

This project is created using Spyder and Visual Studio Code as the IDE for the python and jupyter notebook respectively. The packages used in this project are Pandas, Scikit-learn, TensorFlow Keras and Matplotlib.

## 3. Methodology

### 3.1 Data pipeline

The data is first loaded and preprocessed to properly split them into features and labels. Then the data is split into train and test sets, with a ratio of 80:20.

### 3.2 Model pipeline

A feedforward neural network is constructed that is catered for classification problem. The structure of the model is fairly simple. Figure below shows the structure of the model:

![Model Structure](images/model_1.png)

The model is trained with a batch size of 16 and for 50 epochs. Early stopping and dropout is applied in this training to reduce overfitting. The training stops at epoch 40, with a training accuracy of 96% and validation accuracy of 94%. The results of the training process are shown in the graph visualized by Matplotlib below: 

![Train Acc vs Val Acc](images/train_acc_vs_val_acc.png)
![Train Loss vs Val Loss](images/train_loss_vs_val_loss.png)

These model also recorded TensorBoard logs to observe whether the model overfits or underfits,

To open an embedded tensorboard viewer inside a notebook, copy the following into a code-cell:
```
%tensorboard --logdir images/tb_logs/heart_disease

```

The results recorded in the TensorBoard logs are shown in the images below:

![TB epoch  accuracy](images/tb_epoch_accuracy.png)
![TB epoch loss](images/tb_epoch_loss.png)

## 4. Results

Upon evaluating the model with test data, the model obtain the following test results, as shown in figure below:

![Train Test Results](images/train_test_results.png)

Since both the train and test results have an accuracy above 90%, we can say that the model are accurate.

## Acknowledgements
-  Kah Chun Kong - SHRDC Technical Instructor [![Github](https://img.shields.io/badge/Github-171515?style=flat-square&logo=github&logoColor=black)](https://github.com/ch4mploo/)