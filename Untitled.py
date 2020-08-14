#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from scipy import stats
from sklearn import datasets
from sklearn.cluster import KMeans


# # Load the iris dataset

# In[190]:


iris= datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[191]:


iris_df.head()


# # As it is difficult to visulalise 4D dimension vector space we will try to reduce the dimension to 2D using PCA analysis

# In[192]:


data = iris_df.iloc[:, :]

from sklearn.decomposition import PCA

model= PCA(n_components=2)
pca= model.fit_transform(data)

data['PCA 1']= pca[:,0]
data['PCA 2'] =pca[:,1]


# # visualise the reduced data 

# In[193]:


plt.scatter(data['PCA 1'],data['PCA 2'],c=iris_df.target,cmap='jet')

plt.xlabel('PCA 1', fontsize=14)
plt.ylabel('PCA 2', fontsize=14)


# # Using KMeans to find the cluster **assumed below as k=3**

# In[203]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data)
y_predicted


# # predicting the cluster points

# In[207]:


km.clusters_centers


# # Predicting the cluster using elbow method

# In[204]:


sse =[]
k_range = range(1,10)
optimalk=1
for i in k_range:
    km=KMeans(n_clusters=i)
    km=km.fit(data)
    sse.append(km.inertia_)
    
    if i >1:
        ratio =sse[i-1]/sse[i-2]
        if i< 0.55:
            optimalk=i

sse


# In[211]:


plt.plot(k_range,sse)
plt.title('The elbow method')
plt.xlabel('k')
plt.ylabel('sum of squared errors')


# # the k=3 according to the elbow method

# In[199]:


km=KMeans(n_clusters=3)


# # visualise the predicted data

# In[202]:



plt.scatter(data['PCA 1'].where(data.cluster==0), data['PCA 2'].where(data.cluster==0))
plt.scatter(data['PCA 1'].where(data.cluster==1), data['PCA 2'].where(data.cluster==1))
plt.scatter(data['PCA 1'].where(data.cluster==2), data['PCA 2'].where(data.cluster==2))
plt.legend(targets, loc ='best')


# In[ ]:




