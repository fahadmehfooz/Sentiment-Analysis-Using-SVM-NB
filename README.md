# Sentiment-Analysis-Using-SVM-NaiveBayes-Bagging-Boosting-RandomForest

* Classifying different tweets of users into the categories : Positive and Negative.
* Took a test split of 20%.
* Converting the tokens using a TF-IDF vectorizer.
* Models trained with SVM, NaiveBayes, Bagging with MultinomialNb, AdaBoosting and RandomForest.
* Tuning the hyperprameters of all the models.
* Plotting precision-recall and roc-auc curves.


## Languages Used 
**Python Version:** 3.9.0

## Resources and Tools Used
**Tools:** Jupyter Notebook

**Packages:** Pandas, NumPy, sklearn, nltk, string, and seaborn.  

## Data Used
* **Data taken from kaggle** : https://www.kaggle.com/fahadmehfoooz/sentiment-analysis-svm-nb-bagging-boosting-rf/data

## Data Wrangling and Data Visualization
* Preparing the tweets by removing the stopwords, punctuations and converting it to cleaned text.
* Creating text features using Tf-IDF vectorizer.
* Doing a train test split.
* Plotting precision-recall and roc-auc curves.

![alt text](https://github.com/fahadmehfooz/Sentiment-Analysis-Using-SVM-NB/blob/main/images/__results___20_1.png)

## Model Building 

First, I took a split on the data with training data as 80%. I tried to compare all the models listed down.

Models Used:

* SVM
* NaiveBayes
* Bagging with MultinomialNB as base estimator
* AdaBoost
* RandomForest
* LSTM

## Model performance

**Results:**

![alt text](https://github.com/fahadmehfooz/Sentiment-Analysis-Using-SVM-NB/blob/main/images/accuracies.png)
