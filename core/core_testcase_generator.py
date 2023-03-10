#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

df = pd.read_csv('Vertical_Skills_merged.csv')


# In[8]:


df


# In[10]:


y = list(df)
y


# In[22]:


import random

test_df = pd.DataFrame(columns=['skills','vertical'])

num_cases = 50
num_skills = 15
num_verticals = 10
row_index = 0

for i in range(num_verticals):
    X = df.iloc[0][i]
    skill_list = X.split(",")
    
    for j in range(num_cases):
        test_case_list = []
        for j in range(num_skills):
            test_case_list.append(random.choice(skill_list))
    
        test_case_list = ','.join(map(str, test_case_list)) 
        test_df.loc[row_index] = [test_case_list,y[i]]
        row_index = row_index+1
    
    


# In[25]:


test_df.shape


# In[ ]:




