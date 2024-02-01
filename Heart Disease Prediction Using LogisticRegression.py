#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[28]:


heart_data = pd.read_csv('heart_disease_data.csv')


# In[29]:


heart_data.head()


# In[30]:


heart_data.tail()


# In[31]:


heart_data.shape


# In[32]:


heart_data.info()


# In[33]:


heart_data.isnull().sum()


# In[34]:


heart_data.describe()


# In[35]:


heart_data['target'].value_counts()


# In[36]:


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


# In[37]:


print(X)


# In[38]:


print(Y)


# In[39]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[40]:


print(X.shape, X_train.shape, X_test.shape)


# In[41]:


model = LogisticRegression()


# In[42]:


model.fit(X_train, Y_train)


# In[43]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[44]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[45]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[46]:


print('Accuracy on Test data : ', test_data_accuracy)


# In[47]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# In[50]:


import joblib


# In[57]:


joblib.dump(model,'numpy_pickle_compat')


# In[58]:


model=joblib.load('numpy_pickle_compat')


# In[59]:


model.predict(input_data_reshaped)


# In[ ]:




