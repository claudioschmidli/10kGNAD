<h1><center><a href="https://tblock.github.io/10kGNAD/">Ten Thousand German News Articles Dataset</a></center></h1>
<h2><center>NLP Project work, FHNW Brugg</center></h2>
<h3><center>Base Classifier</center></h3>
<h4><center>Claudio Schmidli</center></h4>
<h4><center>27.11.2023</center></h4>

## Introduction
This project involves training various NLP classification models on the "German news article" dataset, known as the 10k German News Articles Dataset (10kGNAD). This dataset, part of the One Million Posts Corpus, is crucial for German topic text classification and is available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Its significance lies in addressing the lack of comprehensive German language datasets, which are more common in English.

The 10kGNAD is distinctive in German topic classification, featuring 10,273 news articles from an Austrian newspaper, divided into nine topics. German text classification presents unique challenges, such as higher inflection and common long compound words, making it critical to test classifiers on German datasets. The 10kGNAD labels articles using the second part of their topic paths and combines article titles and texts, excluding authors to prevent bias.

The primary goal of this project is to apply and extend the learnings from the class, demonstrating the ability to not only implement NLP models but also to understand and improve their performance through various techniques like data preprocessing, model selection, and hyperparameter tuning. The project serves as a comprehensive example of how to approach and solve a real-world NLP problem in the German language context.


## Project Overview
I have organized my analysis into three Jupyter notebooks, each focusing on a different model and approach:

### [Notebook 1: Exploratory Data Analysis & SGD Classifier](Base_model.ipynb)

In this notebook, I start with an initial exploration of the dataset through explorative data analysis and then proceed to create a base model.
- **Exploratory Data Analysis:** Conducting an in-depth examination of the dataset for better understanding.
- **Data Cleaning and Preprocessing:** Continuing with cleaning and preprocessing the dataset to ensure data quality and analysis suitability.
- **Base Model - SGD Classifier with TF-IDF:** Establishing a benchmark with a simple SGD classifier using TF-IDF word embeddings. The model's performance is evaluated using metrics like accuracy and F1 score, followed by improvements through hyperparameter tuning.
- **Results Analysis:** In this section I analyze the validity of outcomes, misclassified data, and the effectiveness of hyperparameter optimization.

### [Notebook 2: BERT](BERT_model.ipynb)
This notebook focuses on training a pre-trained German BERT model. The process begins with the creation of a custom model head for the classification task and training only these weights while optimizing hyperparameters. Later, I proceed to fine-tune all the model's weights with a conservative learning rate. The model's effectiveness is assessed using the F1 score, an appropriate metric given the dataset's imbalance, and improvements are made through adjustments in model parameters and training methods.

### [Notebook 3: RNN](RNN_model.ipynb)
Here, I work on training a pre-trained RNN using FastText for word embeddings on pre-trained German data. A custom model head is added to the existing RNN model to suit the classification task. The initial focus is on training this custom head, then moving to fine-tuning all layers with a reduced learning rate. Different hyperparameters such as the number of RNN units, the number of RNN layers, dropout rate, activation function, and RNN type are tested.

## Summary of Insights and Model Performance
Due to the imbalance of the dataset the F1 score was used for evaluating the model performances.
- **BERT:** The BERT transformer model shows the best performance, meeting expectations.
- **Base model:** The simpler SGD, TF-IDF-based model shows almost equivalent effectiveness, suggesting its potential as a feasible production model due to its simplicity, speed, and straightforward implementation.
- **RNN:** The RNN, despite initial assumptions, demonstrates commendable performance, indicating its capability in classification tasks despite limited memory for past words.


### Comparative Model Performance

| Model Title                | Model Type                         | F1 Score |
| -------------------------- | ---------------------------------- | -------- |
| TF-IDF/SGD Classifier      | Base Classifier                    | 0.87     |
| Fasttex/RNN, Encoder       | Recurrent Neural Network           | 0.77     |
| BertTokenizer/BERT         | BERT Transformer Model             | 0.89     |


---
*Note: For detailed methodologies, model development, and specific evaluations, please refer to the individual notebooks.*
