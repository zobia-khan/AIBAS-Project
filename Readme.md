# Bank Customer Churn Prediction - AIBAS

---

## Overview

This project is created as part of the course **"M. Grum: Advanced AI-based Application Systems"** by **"Prof. Dr.-Ing. Marcus Grum - Junior Chair for Business Information Science, esp. AI-based Application Systems"** at **"University of Potsdam"**

 This project aims to forecast customer churn using an Artificial Neural Network (ANN) model and compare its performance with an Ordinary Least Squares (OLS) regression model. The dataset is sourced from Kaggle containing over 2,000 records and includes numerical features like CreditScore, Age, Balance, and EstimatedSalary, along with categorical attributes such as Geography, Gender, and Card Type. The target variable, Exited, indicates whether a customer has left the bank. The data is split into 80% training and 20% testing, with the model trained over 20 epochs using a batch size of 32 to optimize predictions. The comparison with OLS helps evaluate the effectiveness of ANN in predicting churn more accurately.

---

## Repository Overview

### code 
Contains scripting files related to scraping and preprocessing dataset and files related to creating ann/ols model and applying those models to activation data.

### data
Contains **".csv"** files, including a preprocesssed dataset, along with separate files for training, testing, and activation.

### images
Contains all dockerfile for building images.

### learningBase and learningBaseOLS
Contains model files along with performance indicators and visualizations for both ann and ols model.

### scenarios
Contains docker-compose files that enable the application of the ann/ols model trained on activation data.

---

## Commands to run the project

- to scrap and process the data: python3.11 code/data_scraping_and_prep/data_scraping.py
data files will be saved to:
data/customer_churn_dataset/joint_data_collection.csv
data/customer_churn_dataset/training_data.csv
data/customer_churn_dataset/test_data.csv
data/customer_churn_dataset/activation_data.csv
- to train ann model: python3.11 code/annRequests/create_annSolution.py
- to train ols model: python3.11 code/olsRequests/create_olsSolution.py

---

## Commands to run Dockerfile


### **learning_base Docker Image**
- **Build the container:**  
  ```sh
docker build -t <DOCKERNAME>/learningbase_bankcustomerchurnprediction -f images/learningBase_BankCustomerChurnPrediction/Dockerfile .


