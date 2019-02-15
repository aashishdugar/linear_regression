
# coding: utf-8
#developed in jupyter notebook.

# In[2]:


get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import sklearn
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston #sklearn comes with datasets to work on.

boston = load_boston()


# In[3]:


boston.keys() #keys to call data


# In[4]:


boston.data #unframed data


# In[5]:


print(boston.feature_names)


# In[6]:


bostondf = pd.DataFrame(boston.data)
bostondf.head()


# In[7]:


bostondf.columns = boston.feature_names #assign feature names to columns
bostondf.head()


# In[30]:


bostondf["PRICE"] = boston.target #add price as y or target


# In[31]:


bostondf.head()


# In[9]:


inp = bostondf.drop('PRICE',axis = 1) #this will serve as input to the model
inp.head()


# In[10]:


from sklearn.linear_model import LinearRegression #apparently you have to import it separately and just "import sklearn" dont work.
lm = LinearRegression()
lm.fit(inp,bostondf.PRICE) #fit the input


# In[11]:


print(lm.coef_) #coefficients of each feature


# In[12]:


lm.intercept_ #point where the line cuts the axis and we get a slope and predictive distinction


# In[13]:


plt.scatter(bostondf.RM,bostondf.PRICE)
plt.xlabel("avg no of rooms")
plt.title("rel. bw rooms and price")
plt.ylabel("price")
plt.show()


# In[14]:


lm.predict(inp)[0:5]


# In[15]:


plt.scatter(bostondf.PRICE,lm.predict(inp))


# In[22]:


mse_full = np.mean((bostondf.PRICE - lm.predict(inp))**2)
print (mse_full)     


# In[ ]:


# The above is a single feature implementation of Linear Regression. In practice, a single feature will give large errors
# so we move on to predict with a cross validation training and testing split


# In[38]:


X_train,X_test,Y_train,Y_test = sklearn.model_selection.train_test_split(inp, bostondf.PRICE, test_size = 0.33, random_state = 5)
boston_xtrain = pd.DataFrame(X_train)
boston_xtest = pd.DataFrame(X_test)
boston_ytrain = pd.DataFrame(Y_train)
boston_ytest = pd.DataFrame(Y_test)
#boston_xtrain.head()
#boston_ytrain.head()
#boston_xtest.head()
#boston_ytest.head()
3#NOTE - cross_validation method has become obsolete. It is now model_selection


# In[41]:


#now to create a new model
lm2 = LinearRegression()
lm2.fit(X_train,Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

#now we get the mean-squared errors
mse_ytrain = np.mean((Y_train - pred_train)**2)
mse_ytest = np.mean((Y_test - pred_test)**2)
print(mse_ytrain)
print(mse_ytest)


# In[52]:


#now plotting scatter
plt.scatter(Y_test,pred_test,c = ['red','blue'])
plt.hlines(y = 20, xmin = 0, xmax = 50)


# In[73]:


#now residual scatter for training data. This gives the difference between the predicted value and the residual value.
plt.scatter(lm2.predict(X_train),(lm2.predict(X_train)-Y_train),c='g',s=40,alpha=0.5) #alpha is transparency,s is scalar or array like
#plt.scatter(lm2.predict(X_test),(lm2.predict(X_test)-Y_test),c='b',s=40,alpha=0.5)
plt.hlines(y=0,xmin=0,xmax=40)
plt.xlabel("Training Prices: $X_i$")
plt.ylabel("Training Price Residuals: $\hat{X}_i$")
plt.title("Training Prices: Normal vs. Residuals")


# In[74]:


plt.scatter(lm2.predict(X_test),(lm2.predict(X_test)-Y_test),c='b',s=40,alpha=0.5)
plt.hlines(y=0,xmin=0,xmax=40)
plt.xlabel("Testing Prices: $X_i$")
plt.ylabel("Testing Price Residuals: $\hat{X}_i$")
plt.title("Testing Prices: Normal vs. Residuals")

