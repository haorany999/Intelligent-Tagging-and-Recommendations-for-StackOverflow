# Intelligent-Tagging-and-Recommendations-for-StackOverflow

## APAN 5430: Applied Text & Natural Language Analytics Term Project

### Group 3
- Sixuan Li
- Wenyang Cao
- Haoran Yang
- Wenling Zhou
- Jake Xiao

### Github Repo
(https://github.com/haorany999/Intelligent-Tagging-and-Recommendations-for-StackOverflow)

## Introduction
This project aims to develop an intelligent tagging and recommendation system for StackOverflow posts using NLP techniques to predict tags and recommend similar posts.

## Data Description
The dataset includes three files: Questions.csv, Answers.csv, and Tags.csv from [Kaggle StackOverflow Data](https://www.kaggle.com/datasets/stackoverflow/stacksample/data).

## Project Steps

### 1. Cosine Similarity Search
- **Notebook**: `results_cosine_sim.ipynb`
- **Tasks**: Implement a search function to find similar posts based on cosine similarity.
- **Techniques**: TF-IDF Vectorization, Cosine Similarity, Spark.

### 2. Tag Prediction Models
- **Notebook**: `results_tag_prediction.ipynb`
- **Tasks**: Train and evaluate different classifiers (SGD, Logistic Regression, XGBoost) for tag prediction.
- **Techniques**: TF-IDF Vectorization, OneVsRestClassifier, Hyperparameter Tuning, Model Evaluation.

### 3. LDA Topic Modeling
- **Notebook**: `results_topic_modeling.ipynb`
- **Tasks**: Train an LDA model to identify hidden topics in the text and visualize them.
- **Techniques**: LDA, Gensim, PyLDAvis.

## Requirements
- numpy
- pandas
- beautifulsoup4
- spacy
- nltk
- scikit-learn
- xgboost
- joblib
- gensim
- pyLDAvis
- matplotlib
- seaborn
- tqdm
- pyspark

## How to Run
1. Clone the repository from Github.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the provided Jupyter notebooks to execute the project steps and train the models.

## Files
- **results_cosine_sim.ipynb**: Cosine similarity search.
- **results_tag_prediction.ipynb**: Tag prediction models.
- **results_topic_modeling.ipynb**: LDA topic modeling.

## License

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=plastic)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)






