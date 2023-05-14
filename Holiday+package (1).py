#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix,plot_confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale


# In[3]:


df=pd.read_csv("D:\Holiday_Package.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe(include='all').T


# In[9]:


df=df.drop('Unnamed: 0',axis=1)
df.head()


# In[10]:


df.isnull().sum()


# In[11]:


dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))


# In[12]:


for feature in df.columns: 
    if df[feature].dtype == 'object': 
        print(feature)
        print(df[feature].value_counts())
        print('\n')


# In[ ]:





# In[13]:


continous_cols = ['Salary','age','educ','no_young_children','no_older_children']
for i in continous_cols:
    sns.boxplot(df[i],whis=2.0)
    plt.grid()
    plt.show();


# In[ ]:





# In[16]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[17]:


lr,ur=remove_outlier(df['Salary'])
df['Salary']=np.where(df['Salary']>ur,ur,df['Salary'])
df['Salary']=np.where(df['Salary']<lr,lr,df['Salary'])


# In[18]:


sns.boxplot(df['Salary'],whis=2.0)
plt.grid()
plt.show()


# In[ ]:





# In[19]:


sns.countplot(df['foreign'])
plt.grid()
plt.show()


# In[20]:


sns.countplot(df['Holliday_Package'])
plt.grid()
plt.show()


# In[ ]:





# In[21]:


sns.boxplot(df['Holliday_Package'],df['Salary'])
plt.grid()
plt.show()


# In[22]:


sns.boxplot(df['Holliday_Package'],df['age'])
plt.grid()
plt.show()


# In[23]:


sns.boxplot(df['Holliday_Package'],df['educ'])
plt.grid()
plt.show()


# In[24]:


sns.boxplot(df['Holliday_Package'],df['no_young_children'])
plt.grid()
plt.show()


# Converting the Target Variable into Categorical

# In[25]:


df['Holliday_Package'] = pd.Categorical(df['Holliday_Package']).codes


# In[26]:


df['Holliday_Package'].value_counts()


# In[27]:


df.info()


# In[28]:


df1 = pd.get_dummies(df, columns=['foreign'],drop_first=True)


# In[29]:


df1.head()


# In[30]:


df1.corr()


# In[31]:


df1.describe().T


# In[32]:


sns.pairplot(df1 ,diag_kind='hist' ,hue='Holliday_Package');


# In[47]:


df1.corr()
plt.figure(figsize=(10,5))
sns.heatmap(df1.corr(), annot=True,mask=np.triu(df1.corr(),+1));


# Split the Data

# In[33]:


X= df1.drop('Holliday_Package',axis=1)
Y=df1['Holliday_Package']


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30 , random_state=1,stratify=df1['Holliday_Package'])


# ## LDA

# In[35]:


model = LogisticRegression(solver='newton-cg',max_iter=10000,penalty='none',verbose=True,n_jobs=2)
model.fit(X_train, Y_train)


# In[36]:


Ytest_predict_prob=model.predict_proba(X_test)
pd.DataFrame(Ytest_predict_prob).head()


# ## Linear Discriminant Analysis

# In[ ]:


clf = LinearDiscriminantAnalysis()
model=clf.fit(X_train,Y_train)


# In[ ]:


# Training Data Class Prediction with a cut-off value of 0.5
pred_class_train = model.predict(X_train)

# Test Data Class Prediction with a cut-off value of 0.5
pred_class_test = model.predict(X_test)


# ## 2.3 Performance Metrics: Check the performance of Predictions on Train and Test sets using Accuracy, Confusion Matrix, Plot ROC curve and get ROC_AUC score for each model Final Model: Compare Both the models and write inference which model is best/optimized.

# ## Logistic Regression Model

# In[37]:


model.score(X_train, Y_train)


# In[38]:


# predict probabilities
probs = model.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_train, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(Y_train, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr);


# In[39]:


model.score(X_test, Y_test)


# In[40]:


# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
test_auc = roc_auc_score(Y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(Y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr);


# ## Predicting on Training and Test dataset

# In[42]:


Ytrain_predict = model.predict(X_train)
Ytest_predict = model.predict(X_test)


# ## Confusion Matrix for the training data

# In[43]:


confusion_matrix(Y_train, Ytrain_predict)


# In[44]:


print(classification_report(Y_train, Ytrain_predict))


# ## Confusion Matrix for test data

# In[45]:


confusion_matrix(Y_test, Ytest_predict)


# In[46]:


print(classification_report(Y_test, Ytest_predict))


# In[50]:


auc = metrics.roc_auc_score(Y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(Y_train,pred_prob_train[:,1])
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'Training Data')


# AUC and ROC for the test data

# calculate AUC
auc = metrics.roc_auc_score(Y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(Y_test,pred_prob_test[:,1])
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Test Data')
# show the plot
plt.legend(loc='best')
plt.show()


# In[49]:


# Training Data Probability Prediction
pred_prob_train = model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = model.predict_proba(X_test)
pred_prob_train[:,1]


# In[ ]:




