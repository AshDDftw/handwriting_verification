# Handwriting Verification ML Project

This project aims to develop a machine learning model for handwriting verification. The model is trained to distinguish between genuine and forged handwriting samples based on a dataset of handwritten samples.

## Project Overview

The handwriting verification ML project focuses on the following key aspects:

- Dataset collection and preprocessing: The project involves the collection of a handwriting dataset, including genuine and forged samples. The dataset is preprocessed to extract relevant features and prepare it for training.

- Model development: A machine learning model is built using a siamese model.
- Model evaluation: The trained model is evaluated using appropriate evaluation metrics to assess its performance in distinguishing between genuine and forged handwriting samples. Various metrics, such as auc, precision, recall, and F1 score, are considered.

- Inference and deployment: The trained model is utilized to perform handwriting verification on new, unseen samples.

## Project Structure

The project contains the following :

`handwriting_verification_training.ipynb`:

  - `create_model`: creation,complilation and training of the model
  
  - `make_paired_dataset`: making paired image dataset and corresponding labels for training and validation purposes
  
  - `createDataset`: Creation of training and validation datasets
  
  - `validate_model`:validating the model
  
  - `create_new_model`:to create a new model , train and validate the trained model

`handwriting_verification_inference.ipynb`

  - `inference.py`: Python script for performing handwriting verification on new samples using the trained model.
  
  - `test_predefined_model`:to test a saved model
  
`README.md`: Project documentation explaining the project overview, structure, and usage instructions.
  

Follow the instructions to preprocess the dataset, train the model, evaluate its performance, and perform inference.

-to create and train a new model run `create_new_model(batch_size,epochs,train_dataset_path,validation_file_path,,validation_images_file,train_size,val_size)` where  
1.batch_size- size of batch size for the siamese model  
2.epochs - number of epochs  
3.train_dataset_path- path of dataset containing the folders of different writers and each containing the image of their handwriting  
4.validation_csv_path- path for csv file of validation file  
5.validation_images_file - folder containing images of validation images  
6.train_size- number of rows in training data  
7.val_size- number of rows in validation set(during model training)  

-to test a pre-trained model run `test_predefined_model(model_path,test_path,test_images_path)` where  
1.model_path- path of saved model (tensorflow model)  
2.test_csv_path- path of csv file of test images' path  
3.test_images_path- folder path of test images  

## Google drive for tarining and inference data and model  
[Drive Link](https://drive.google.com/drive/folders/1SFyf2ETj2wguzD_cORtO2OzAiNAHDv8h?usp=sharing)
