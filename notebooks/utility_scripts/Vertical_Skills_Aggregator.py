#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import glob


# In[ ]:


from collections import OrderedDict


# ### Reading list of Skills for each Vertical from Naukri Dataset

# #### Loading all the files

# In[ ]:


directory_path =  os. getcwd() 
os.chdir(directory_path + "/data/Naukri_Data/")
file_list = glob.glob( "*.xlsx")


# In[ ]:


jobs_df = []

for file in file_list:
  jobs_df.append(pd.read_excel(file))

df = pd.concat(jobs_df, sort=False, ignore_index=True)
del jobs_df

os.chdir(directory_path)


# In[ ]:


Skills_df = df.groupby(['Vertical'])['Skills_List'].apply(','.join).reset_index()


# ### check all the skills keyword and prepare seeds

# In[ ]:



Skills_df.fillna('', inplace=True)
Skills_df['Skills_List']  = Skills_df['Skills_List'].str.lower()


# In[ ]:


Skills_df.head(10)


# In[ ]:


skills = Skills_df.Skills_List.tolist()
skills = list(filter(None, skills))


# In[ ]:


Skills_df['Unique']=''


# In[ ]:


for i in range(len(skills)) :
    Skills_df['Unique'][i] = ','.join(OrderedDict.fromkeys(Skills_df['Skills_List'][i].split(',')))


# In[ ]:


Skills_df.drop(['Skills_List'], axis =1, inplace = True)


# ### Creating a csv with the Seed List of Skills for all Verticals

# In[ ]:


Vert_Skills = Skills_df.transpose()


# In[ ]:


new_header = Vert_Skills.iloc[0]
Vert_Skills = Vert_Skills[1:]
Vert_Skills.columns = new_header


# In[ ]:


Vert_Skills.to_csv('Vertical_Skills_Naukri.csv', index = False)


# ### Loading the list prepared from ChatGPT Output 

# In[ ]:


Manual_Skills_DF = pd.read_excel('./data/Skills_Per_vertical_Dict_ChatGPT.xlsx')


# In[ ]:


cols = Manual_Skills_DF.columns


# In[ ]:


for vert in cols :
    Manual_Skills_DF[vert]  = Manual_Skills_DF[vert].str.lower()


# In[ ]:


Skills_ListM = Manual_Skills_DF.apply(','.join)
Skills_ListM = pd.DataFrame(Skills_ListM)


# In[ ]:


Skills_ListM_T = Skills_ListM.transpose()


# In[ ]:


Skills_ListM_T.to_csv('Vertical_Skills_ChatGPT.csv', index = False)


# ### Merging both the Skill Lists

# In[ ]:


Skills_MN = pd.concat([Vert_Skills,Skills_ListM_T], axis=0)


# In[ ]:


Skills_MN = Skills_MN.apply(','.join)


# In[ ]:


Skills_MN = pd.DataFrame(Skills_MN)


# In[ ]:


Skills_MNT = Skills_MN.transpose()


# In[ ]:


cols = Skills_MNT.columns


# In[ ]:


for vert in cols :
    Skills_MNT[vert][0] = ','.join(OrderedDict.fromkeys(Skills_MNT[vert][0].split(',')))


# #### Creating a List for each Vertical - Sample code

# In[ ]:


Cloud_Computing_L = Skills_MNT['Cloud-Computing'].tolist()


# In[ ]:


Cloud_Computing_List = Cloud_Computing_L[0].split(",")


# ### Generate the file that has Merged Skills List 

# In[ ]:


Skills_MNT.to_csv('Vertical_Skills_merged.csv', index = False)


# ### Display all the Key Words for each Vertical 

# In[ ]:


for vert in cols :
    skills_list  = Skills_MNT[vert].tolist()
    skills_split = skills_list[0].split(",")
    print ("********** ", vert , "**********")
    print(skills_split)
    print ("***********************************************************************************")

