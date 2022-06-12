#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pyspark
pyspark.__version__


# In[70]:


import findspark
findspark.init()
findspark.find()


# In[71]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Adult_Data").getOrCreate() 
spark


# In[72]:


authors = spark.read.csv('ratings.csv', sep=',', inferSchema=True, header=True)


# In[73]:



authors.show()


# In[74]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


# In[75]:


df=authors.rdd.map(lambda x: [x[3],x[1]*20/100])


# In[76]:


print(df)


# In[ ]:



als  =  ALS ( maxIter = 5 ,  regParam = 0.01 ,  userCol = "userId" ,  itemCol = "movieId" ,  ratingCol = "rating" , 
          coldStartStrategy = "drop" ) 
model  =  als . adapté (training ) 


# In[ ]:


prédictions  =  modèle . transform ( test ) 
evaluator  =  RegressionEvaluator ( metricName = "rmse" ,  labelCol = "rating" , 
                                predictionCol = "prediction" ) 
rmse  =  evaluator . évaluer ( prédictions ) 
imprimer ( "Erreur quadratique moyenne = "  +  str ( rmse ))


# In[ ]:



als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,
          userCol="userId", itemCol="movieId", ratingCol="rating")

