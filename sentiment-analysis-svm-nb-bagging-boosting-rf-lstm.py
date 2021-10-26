#!/usr/bin/env python
# coding: utf-8

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Importing Libraries </h1>
# </div>
# 

# In[37]:


import pandas as pd
import seaborn as sns
import re, nltk
nltk.download('punkt')
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import f1_score
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout


# In[38]:


df =pd.read_csv(r'../input/twitter-airline-sentiment/Tweets.csv')
df.head()


# 

# In[39]:


# Unique values of sentiment
df['airline_sentiment'].unique()


# In[40]:


# Unique values of sentiment plot

ax = sns.countplot(x="airline_sentiment", data=df)


# > Positive and neutral tweets are almost equal.
# 
# > Negative tweets are more than double of neutral or positive sentiments.

# In[41]:


# Unique values of airline

plt.figure(figsize=(10,10))
ax = sns.countplot(x="airline", data=df)


# > United has the most number of flights.
# 
# > Virgin America has the least.

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Utility Functions </h1>
# </div>
# 

# In[42]:


# I am tokenizing the tweet and also taking tokens from second index onwards as initital to gives airline name and '@' and lowering thm and later making it back a sentence
def clean_the_tweet(text):
  tokens= nltk.word_tokenize(re.sub("[^a-zA-Z]", " ",text))
  tokens = [token.lower() for token in tokens]
  return ' '.join(tokens[2:])

                 

def text_process(msg):
  nopunc =[char for char in msg if char not in string.punctuation]
  nopunc=''.join(nopunc)
  return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


 
def check_scores(clf,X_train, X_test, y_train, y_test):

  model=clf.fit(X_train, y_train)
  predicted_class=model.predict(X_test)
  predicted_class_train=model.predict(X_train)
  test_probs = model.predict_proba(X_test)
  test_probs = test_probs[:, 1]
  yhat = model.predict(X_test)
  lr_precision, lr_recall, _ = precision_recall_curve(y_test, test_probs)
  lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)


  print('Train confusion matrix is: ',)
  print(confusion_matrix(y_train, predicted_class_train))

  print()
  print('Test confusion matrix is: ')
  print(confusion_matrix(y_test, predicted_class))
  print()
  print(classification_report(y_test,predicted_class)) 
  print() 
  train_accuracy = accuracy_score(y_train,predicted_class_train)
  test_accuracy = accuracy_score(y_test,predicted_class)

  print("Train accuracy score: ", train_accuracy)
  print("Test accuracy score: ",test_accuracy )
  print()
  train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
  test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

  print("Train ROC-AUC score: ", train_auc)
  print("Test ROC-AUC score: ", test_auc)
  fig, (ax1, ax2) = plt.subplots(1, 2)

  ax1.plot(lr_recall, lr_precision)
  ax1.set(xlabel="Recall", ylabel="Precision")

  plt.subplots_adjust(left=0.5,
                    bottom=0.1, 
                    right=1.5, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
  print()
  print('Are under Precision-Recall curve:', lr_f1)
  
  fpr, tpr, _ = roc_curve(y_test, test_probs)


  ax2.plot(fpr, tpr)
  ax2.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

  print("Area under ROC-AUC:", lr_auc)
  return train_accuracy, test_accuracy, train_auc, test_auc



def grid_search(model, parameters, X_train, Y_train):
  #Doing a grid
  grid = GridSearchCV(estimator=model,
                       param_grid = parameters,
                       cv = 2, verbose=2, scoring='roc_auc')
  #Fitting the grid 
  grid.fit(X_train,Y_train)
  print()
  print()
  # Best model found using grid search
  optimal_model = grid.best_estimator_
  print('Best parameters are: ')
  print( grid.best_params_)

  return optimal_model
  


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Text Preparation </h1>
# </div>
# 

# In[43]:


# removing neutral tweets

df = df[df['airline_sentiment']!='neutral']
df['cleaned_tweet'] = df['text'].apply(clean_the_tweet)

df.head()
df['airline_sentiment'] = df['airline_sentiment'].apply(lambda x: 1 if x =='positive' else 0)
df.head()


# In[44]:


# Cleaning the tweets, removing punctuation marks
df['cleaned_tweet'] = df['cleaned_tweet'].apply(text_process)
df.reset_index(drop=True, inplace = True)
df.head()


# In[45]:


df['airline_sentiment'].unique()


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Base SVM model with TF-IDF </h1>
# </div>
# 

# In[46]:


# Creating object of TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True)
X_tf_idf= vectorizer.fit_transform(df.cleaned_tweet)
x_train, x_test, y_train, y_test = train_test_split(X_tf_idf, df['airline_sentiment'], random_state=42)


# 

# 

# In[47]:



SVM = svm.SVC( probability=True)
s_train_accuracy, s_test_accuracy, s_train_auc, s_test_auc = check_scores(SVM,x_train, x_test, y_train, y_test)


# > With increase in FPR, TPR also increases.
# 
# > With increase in recall, precision decreases.

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> After optimizing the hyperparameters with TF-IDF </h1>
# </div>
# 

# In[48]:


# Tuning the hyperparameters
parameters ={
    "C":[0.1,1,10],
    "kernel":['linear', 'rbf', 'sigmoid'],
    "gamma":['scale', 'auto']
}



svm_optimal = grid_search(svm.SVC(probability=True), parameters,x_train, y_train)


# In[49]:


so_train_accuracy, so_test_accuracy, so_train_auc, so_test_auc = check_scores(svm_optimal,x_train, x_test, y_train, y_test)


# > With increase in recall, preciison decreases which makes sense also.
# 
# > With increase in TPR, FPR inceases.

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Using Multinomial Naive Bayes </h1>
# </div>
# 

# In[50]:


m_train_accuracy, m_test_accuracy, m_train_auc, m_test_auc = check_scores(MultinomialNB(),x_train, x_test, y_train, y_test)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Using Gaussian Naive Bayes
#  </h1>
# </div>
# 

# In[51]:


g_train_accuracy, g_test_accuracy, g_train_auc, g_test_auc=check_scores(GaussianNB(),x_train.toarray(), x_test.toarray(), y_train, y_test)


# > It is interesting to see in Naive Bayes, we are getting linear relationship.

# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Using AdaBoost
#  </h1>
# </div>
# 

# > **base estimator here is: decision stump**

# In[52]:


a_train_accuracy, a_test_accuracy, a_train_auc, a_test_auc=check_scores(AdaBoostClassifier(),x_train,x_test, y_train, y_test)


# In[53]:


params = {'n_estimators': [10, 50, 100, 500],
 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
 'algorithm': ['SAMME', 'SAMME.R']}

ada_optimal_model = grid_search(AdaBoostClassifier(), params,x_train, y_train)


# In[54]:


ao_train_accuracy, ao_test_accuracy, ao_train_auc, ao_test_auc=check_scores(ada_optimal_model,x_train,x_test, y_train, y_test)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold"> Bagging with MultinomialNB
#  </h1>
# </div>
# 

# In[55]:




kfold = model_selection.KFold(n_splits = 3)
  
# bagging classifier
model = BaggingClassifier(base_estimator = MultinomialNB(),
                          n_estimators = 100)

b_train_accuracy, b_test_accuracy, b_train_auc, b_test_auc= check_scores(model,x_train,x_test, y_train, y_test)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold">Using Random Forest
#  </h1>
# </div>
# 
# 

# In[56]:


r_train_accuracy, r_test_accuracy, r_train_auc, r_test_auc= check_scores(RandomForestClassifier(random_state=0).fit(x_train, y_train), x_train,x_test,y_train,y_test)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold">Using LSTM
#  </h1>
# </div>
# 
# 

# In[57]:


corpus = [df['cleaned_tweet'][i] for i in range( len(df))]

voc_size=5000

onehot_=[one_hot(words,voc_size)for words in corpus] 

max_sent_length=max([len(i) for i in corpus])

embedded_docs=pad_sequences(onehot_,padding='pre',maxlen=max_sent_length)
    
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=max_sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

X_final=np.array(embedded_docs)
y_final=np.array(df['airline_sentiment'])
X_final.shape,y_final.shape


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[59]:


y_test_pred=model.predict_classes(X_test)
y_train_pred=model.predict_classes(X_train)


# In[60]:


test_acc_lstm = accuracy_score(y_test,y_test_pred)
train_acc_lstm = accuracy_score(y_train,y_train_pred)
test_roc_lstm = roc_auc_score(y_test,y_test_pred)
train_roc_lstm = roc_auc_score(y_train,y_train_pred)


# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold">Final Results
#  </h1>
# </div>
# 
# 

# In[61]:


data = [('Random Forest', r_train_accuracy, r_test_accuracy, r_train_auc, r_test_auc),
 ('MultinomialNB',m_train_accuracy, m_test_accuracy, m_train_auc, m_test_auc  ),
('Bagged MultinomialNB',b_train_accuracy, b_test_accuracy, b_train_auc, b_test_auc ),
 ('AdaBoost',a_train_accuracy, a_test_accuracy, a_train_auc, a_test_auc ),
('AdaBoost Optimized',ao_train_accuracy, ao_test_accuracy, ao_train_auc, ao_test_auc),
('Gaussian Naive Bayes',g_train_accuracy, g_test_accuracy, g_train_auc, g_test_auc),
('SVM', s_train_accuracy, s_test_accuracy, s_train_auc, s_test_auc),
('SVM Optimized', so_train_accuracy, so_test_accuracy, so_train_auc, so_test_auc),
('LSTM',train_acc_lstm, test_acc_lstm, train_roc_lstm, test_roc_lstm )]


Scores_ =pd.DataFrame(data = data, columns=['Model Name','Train Accuracy', 'Test Accuracy', 'Train ROC', 'Test ROC'])
Scores_.set_index('Model Name', inplace = True)

Scores_


# 
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            font-size:110%;
#            font-family:cursive;
#            letter-spacing:0.5px;
#            background-color:powderblue;
#            color:Black;
#            font-family:cursive
#            ">
# <h1 style="text-align:center;font-weight: bold">Conclusion
#  </h1>
# </div>
# 
# **Most of the models are doing pretty well here.**
