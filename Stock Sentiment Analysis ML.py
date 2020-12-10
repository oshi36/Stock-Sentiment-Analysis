#!/usr/bin/env python
# coding: utf-8

# # Stock Sentiment Analysis From News Headlines In ML

# In[70]:


# Dataset : https://www.kaggle.com/aaron7sun/stocknews


# In[71]:


import pandas as pd
import numpy as np
df=pd.read_csv('./data/Combined_News_DJIA.csv' , encoding="ISO-8859-1")
df.head()


# In[72]:


print("Shape of Dataset is :" ,df.shape)
print("Dimension of Dataset is :" ,df.ndim)
print("Size/Total Data in Dataset is :" ,df.size)


# In[73]:


print(df.index)
print("--------------------------------------------------------------------------")
print(df.columns)


# In[74]:


df.tail()


# In[75]:


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[76]:


# Removing Punctuations
data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True , inplace=True)

data.replace("b"," ",regex=True , inplace=True)

# Renaming column names
list1 = [ i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
data.head(5)


# In[77]:


# converting headlines to lower case
for index in new_Index :
    data[index] = data[index].str.lower()
data.head(1)


# In[78]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[79]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[80]:


headlines[0]


# In[81]:


# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[82]:


# Implement BAG OF WORDS
countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)


# In[83]:


traindataset[0]


# In[84]:


# Implement RandomForestClassifier
randomclassifier = RandomForestClassifier(n_estimators = 200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[85]:


# Predict For the Test Dataset
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[86]:


test.loc[1611,:]


# In[87]:


predictions


# In[88]:


# Import Library To Check Accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[89]:


matrix = confusion_matrix(test['Label'],predictions)
print(matrix)
score = accuracy_score(test['Label'],predictions)
print(score)
report = classification_report(test['Label'],predictions)
print(report)


# In[90]:


# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[91]:


# Implement BAG OF WORDS
tfidfvector = TfidfVectorizer(ngram_range=(2,2))
traindataset = tfidfvector.fit_transform(headlines)


# In[92]:


# Implement RandomForestClassifier
randomclassifier = RandomForestClassifier(n_estimators = 200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[93]:


# Predict For the Test Dataset
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[94]:


predictions


# In[95]:


matrix = confusion_matrix(test['Label'],predictions)
print(matrix)
score = accuracy_score(test['Label'],predictions)
print(score)
report = classification_report(test['Label'],predictions)
print(report)


# In[96]:


traindataset


# In[97]:


# Implement Naive Bayes
from sklearn.naive_bayes import MultinomialNB
naive = MultinomialNB()
naive.fit(traindataset,train['Label'])


# In[98]:


naive


# In[99]:


# Predict For the Test Dataset
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = tfidfvector.transform(test_transform)
predictions = naive.predict(test_dataset)


# In[100]:


predictions


# In[101]:


matrix = confusion_matrix(test['Label'],predictions)
print(matrix)
score = accuracy_score(test['Label'],predictions)
print(score)
report = classification_report(test['Label'],predictions)
print(report)


# In[ ]:




