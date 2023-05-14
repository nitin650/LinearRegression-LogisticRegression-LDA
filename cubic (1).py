#!/usr/bin/env python
# coding: utf-8

# In[398]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[399]:


df=pd.read_csv("D:\cubic_zirconia.csv")


# In[400]:


df.head(10)


# In[401]:


df.groupby('carat').sum()


# In[402]:


df=df.drop('Unnamed: 0',axis=1)


# In[403]:


df.duplicated().sum()


# In[404]:


df.isnull().sum()


# In[405]:


for column in df.columns:
    if df[column].dtype=='float':
        median=df[column].median()
        df[column]=df[column].fillna(median)


# In[406]:


df.isnull().sum()


# In[407]:


df.duplicated().sum()


# In[408]:


df=df.drop_duplicates()


# In[409]:


df.duplicated().sum()


# In[410]:


df.info()


# In[411]:


for column in df.columns:
    if df[column].dtype == 'object':
        print(column.upper(),': ',df[column].nunique())
        print(df[column].value_counts().sort_values())
        print('\n')


# In[412]:


cont_cols = ['carat','depth','table','x','y','z','price']
for i in cont_cols:
    sns.boxplot(df[i],whis=1.5)
    plt.grid()
    plt.show();


# In[413]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lr=Q1-(1.5*IQR)
    ur=Q3+(1.5*IQR)
    return lr,ur


# In[414]:


lr,ur=remove_outlier(df['carat'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['carat']=np.where(df['carat']>ur,ur,df['carat'])
df['carat']=np.where(df['carat']<lr,lr,df['carat'])
print('')

lr,ur=remove_outlier(df['depth'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['depth']=np.where(df['depth']>ur,ur,df['depth'])
df['depth']=np.where(df['depth']<lr,lr,df['depth'])
print('')

lr,ur=remove_outlier(df['table'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['table']=np.where(df['table']>ur,ur,df['table'])
df['table']=np.where(df['table']<lr,lr,df['table'])
print('')

lr,ur=remove_outlier(df['x'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['x']=np.where(df['x']>ur,ur,df['x'])
df['x']=np.where(df['x']<lr,lr,df['x'])
print('')

lr,ur=remove_outlier(df['y'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['y']=np.where(df['y']>ur,ur,df['y'])
df['y']=np.where(df['y']<lr,lr,df['y'])
print('')

lr,ur=remove_outlier(df['z'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['z']=np.where(df['z']>ur,ur,df['z'])
df['z']=np.where(df['z']<lr,lr,df['z'])
print('')

lr,ur=remove_outlier(df['price'])
print('Lower Range :',lr,'\nUpper Range :',ur)
df['price']=np.where(df['price']>ur,ur,df['price'])
df['price']=np.where(df['price']<lr,lr,df['price'])
print('')


# In[415]:


cont_cols = ['carat','depth','table','x','y','z','price']
for i in cont_cols:
    sns.boxplot(df[i],whis=1.5)
    plt.grid()
    plt.show();


# In[416]:


sns.countplot(df['cut'])


# In[417]:


sns.countplot(df['color'])


# In[418]:


sns.countplot(df['clarity'])


# In[419]:


sns.countplot(df['color'])


# In[420]:


sns.pairplot(df[cont_cols])


# In[421]:


corr=df.corr()


# In[422]:


plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,fmt='.2g',)


# ## converting all objects into catagorical
# 

# In[423]:


df['cut']=np.where(df['cut']=='Ideal','2',df['cut'])
df['cut']=np.where(df['cut']=='Premium','2',df['cut'])

df['cut']=np.where(df['cut']=='Good','1',df['cut'])
df['cut']=np.where(df['cut']=='Very Good','1',df['cut'])

df['cut']=np.where(df['cut']=='Fair','0',df['cut'])


# In[424]:


df['clarity'].value_counts()


# In[425]:


df['clarity']=np.where(df['clarity']=='I1','Best',df['clarity'])

df['clarity']=np.where(df['clarity']=='SI1','VGood',df['clarity'])
df['clarity']=np.where(df['clarity']=='SI2','VGood',df['clarity'])

df['clarity']=np.where(df['clarity']=='VS1','Good',df['clarity'])
df['clarity']=np.where(df['clarity']=='VS2','Good',df['clarity'])


df['clarity']=np.where(df['clarity']=='VVS1','Bad',df['clarity'])
df['clarity']=np.where(df['clarity']=='VVS2','Bad',df['clarity'])

df['clarity']=np.where(df['clarity']=='IF','Worst',df['clarity'])


# In[426]:


df.head()


# In[427]:


df['color']=np.where(df['color']=='J','Best',df['color'])

df['color']=np.where(df['color']=='I','VGood',df['color'])
df['color']=np.where(df['color']=='H','VGood',df['color'])

df['color']=np.where(df['color']=='G','Good',df['color'])
df['color']=np.where(df['color']=='F','Good',df['color'])

df['color']=np.where(df['color']=='E','Bad',df['color'])

df['color']=np.where(df['color']=='D','Worst',df['color'])


# In[428]:


df.head()


# In[429]:


df['clarity']=np.where(df['clarity']=='Best','4',df['clarity'])

df['clarity']=np.where(df['clarity']=='VGood','3',df['clarity'])
df['clarity']=np.where(df['clarity']=='VGood','3',df['clarity'])

df['clarity']=np.where(df['clarity']=='Good','2',df['clarity'])
df['clarity']=np.where(df['clarity']=='Good','2',df['clarity'])

df['clarity']=np.where(df['clarity']=='Bad','1',df['clarity'])
df['clarity']=np.where(df['clarity']=='Bad','1',df['clarity'])

df['clarity']=np.where(df['clarity']=='Worst','0',df['clarity'])


# In[430]:


df['color']=np.where(df['color']=='Best','4',df['color'])

df['color']=np.where(df['color']=='VGood','3',df['color'])
df['color']=np.where(df['color']=='VGood','3',df['color'])

df['color']=np.where(df['color']=='Good','2',df['color'])
df['color']=np.where(df['color']=='Good','2',df['color'])

df['color']=np.where(df['color']=='Bad','1',df['color'])

df['color']=np.where(df['color']=='Worst','0',df['color'])


# In[431]:


df.head()


# In[432]:


df.info()


# In[433]:


df['cut']=df['cut'].astype('int64')
df['color']=df['color'].astype('int64')
df['clarity']=df['clarity'].astype('int64')


# In[434]:


df.info()


# In[435]:


df_dummy=pd.get_dummies(df,columns=['clarity'])


# In[436]:


df_dummy.shape


# ## 1.3 Encode the data (having string values) for Modelling. Split the data into train and test (70:30). Apply Linear regression using scikit learn. Perform checks for significant variables using appropriate method from statsmodel. Create multiple models and check the performance of Predictions on Train and Test sets using Rsquare, RMSE & Adj Rsquare. Compare these models and select the best one with appropriate reasoning.

# In[437]:


x=df_dummy.drop('price',axis=1)
y=df_dummy.pop('price')


# In[438]:


from sklearn.model_selection import train_test_split


# In[439]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


# In[ ]:





# In[440]:


regression_model = LinearRegression()
regression_model.fit(x_train, y_train)


# In[443]:


regression_model.score(x_train, y_train)


# In[444]:


regression_model.score(x_test, y_test)


# In[445]:


predicted_train=regression_model.fit(x_train, y_train).predict(x_train)
np.sqrt(metrics.mean_squared_error(y_train,predicted_train))


# In[446]:


predicted_test=regression_model.fit(x_train, y_train).predict(x_test)
np.sqrt(metrics.mean_squared_error(y_test,predicted_test))


# In[447]:


data_train = pd.concat([x_train, y_train], axis=1)
data_test=pd.concat([x_test,y_test],axis=1)
data_train.head()


# In[448]:


data_train.columns


# In[449]:


expr= 'price ~ carat+ cut+ color +depth + table+ x+ y+ z+ clarity_0+ clarity_1 + clarity_2+ clarity_3 + clarity_4'


# In[450]:


import statsmodels.formula.api as smf


# In[451]:


lm1 = smf.ols(formula= expr, data = data_train).fit()
lm1.params


# In[452]:


print(lm1.summary())


# In[453]:


# Calculate MSE
mse = np.mean((lm1.predict(data_train.drop('price',axis=1))-data_train['price'])**2)


# In[454]:


np.sqrt(mse)


# In[455]:


np.sqrt(lm1.mse_resid) 


# In[456]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[457]:


vif = [variance_inflation_factor(x.values, ix) for ix in range(x.shape[1])]


# In[458]:


i=0
for column in x.columns:
    if i < 14:
        print (column ,"--->",  vif[i])
        i = i+1


# In[459]:


for i,j in np.array(lm1.params.reset_index()):
    print('({}) * {} +'.format(round(j,2),i),end=' ')


# In[ ]:





# In[ ]:





# In[ ]:




