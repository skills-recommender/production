#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import glob
directory_path = os.getcwd()
df = pd.DataFrame()


# ### Reading the Candidate Profiles Data

# In[3]:


os.chdir(directory_path + "/../Input_Data/Candidates")


# In[4]:


df = pd.read_csv('Candidate_Profile.csv', nrows=1100000, engine='python', sep='","', names=range(11), header=None)


# ### Doing basic data cleanup for Uniform format 

# In[5]:


df.columns=["experience_id", "company", "description", "end_date","location", "start_date", "title", "user_id","UN1","UN2","UN3"]
 


# In[6]:


df['experience_id'] = df.experience_id.replace('"','', regex=True)
df['user_id'] = df.user_id.replace('"','', regex=True)


# In[7]:


df_UN1 = df.loc[df['UN1'].notnull()]


# In[8]:


drop_list = df[df['UN1'].notnull()].index.tolist()


# In[9]:


df.drop(index=drop_list, axis=0, inplace = True)


# In[10]:


df.drop(['UN1','UN2','UN3'], axis =1, inplace = True)


# ### Count the number of Users who have less than 5 entries

# In[12]:


sub_df_less_5 = df.groupby('user_id').filter(lambda x : len(x)<5)


# ## Picking the entries of those users who have a minimum of 5 entries 
# 
# ### Under the assumption that these users are likely to have a detailed resume with more skill keywords.

# In[14]:


DResume_df = df[df.groupby('user_id').user_id.transform('count')>4].copy()


# ### Individual Contributor Role

# In[95]:


df_IC = DResume_df[DResume_df["title"].str.contains("Manager|manager|Director|VP|SVP|Vice President|Head of|Head")  == False]


# ### Manager / Leadership Role Profiles

# In[17]:


df_Mgr = DResume_df[DResume_df["title"].str.contains("Manager|manager|Director|VP|SVP|Vice President|Head of|Head")  == True]


# ### Create seperate files - Role Wise

# In[19]:


df_Mgr.to_csv('Candidates_Mgr_profiles.csv', index = False)


# In[20]:


df_IC.to_csv('Candidates_IC_profiles.csv', index = False)


# ### Mgr Profiles - Grouped by User_ID

# In[57]:


Mgr_Desc = df_Mgr[['description','user_id','company']].copy()


# In[58]:


Mgr_Desc['user_id'] = Mgr_Desc['user_id'].astype(int)
Mgr_Desc['description'] = Mgr_Desc['description'].astype(str)
Mgr_Desc['company'] = Mgr_Desc['company'].astype(str)


# In[59]:


Mgr_Desc_Grp_Result = Mgr_Desc.groupby('user_id', as_index=False).agg({'user_id' : 'first', 'description' : ','.join, 'company' : ','.join, })


# In[60]:


Mgr_Desc_Grp_Result


# In[61]:


Mgr_Desc_Grp_Result.to_csv('Mgr_profiles_grouped_by_userid.csv', index = False)


# ### IC Profiles - Grouped by User_ID

# #### Skipping Profiles related to Sales, Marketing, Finance, HR, Teaching

# In[97]:


df_IC_Tech = df_IC[df_IC["title"].str.contains("Sales|sales|Marketing|marketing|Business|Chief of Staff|Finance|Financial|School|Merchandising|Tutor|Professor|Founder|Talent|Acquisition|Hiring|Recruitment|Placement")  == False]


# In[98]:


IC_Desc = df_IC_Tech[['description','user_id','company', 'title']].copy()
IC_Desc = IC_Desc.drop(IC_Desc.index[0])


# In[99]:


IC_Desc['user_id'] = IC_Desc['user_id'].astype(int)
IC_Desc['description'] = IC_Desc['description'].astype(str)
IC_Desc['company'] = IC_Desc['company'].astype(str)
IC_Desc['title'] = IC_Desc['title'].astype(str)


# In[100]:


IC_Desc_Grp_Result = IC_Desc.groupby('user_id', as_index=False).agg({'user_id' : 'first', 'description' : ','.join, 'company' : ','.join,  'title' : ','.join })


# In[101]:


IC_Desc_Grp_Result.info()


# In[102]:


IC_Desc_Grp_Result.to_csv('IC_profiles_grouped_by_userid.csv', index = False)


# ## Filtering based on Top IT Companies - to get Resumes related to IT Industry

# In[103]:


df_IC_Tech_Top5 = IC_Desc_Grp_Result[IC_Desc_Grp_Result["company"].str.contains("Microsoft|Infosys|Wipro|Google|Accenture")  == True]


# In[104]:


df_IC_Tech_Top5.info()


# In[107]:


df_IC_Tech_Top5 = df_IC_Tech_Top5[df_IC_Tech_Top5["description"].str.contains("None|none|NONE")  == False]


# In[109]:


df_IC_Tech_Top5 =  df_IC_Tech_Top5[df_IC_Tech_Top5["title"].str.contains("Software Engineer|Senior Software Engineer")  == True]


# In[110]:


df_IC_Tech_Top5.info()


# ### Creating files with multiple samples

# In[111]:


Sample1 = df_IC_Tech_Top5_filt.sample(n = 200)


# In[112]:


Sample2 = df_IC_Tech_Top5_filt.sample(n = 200)


# In[113]:


Sample3 = df_IC_Tech_Top5_filt.sample(n = 200)


# In[114]:


Sample1.to_csv('IC_IT_Tech_200_RandomSet_4.csv', index = False)


# In[115]:


Sample2.to_csv('IC_IT_Tech_200_RandomSet_5.csv', index = False)


# In[116]:


Sample3.to_csv('IC_IT_Tech_200_RandomSet_6.csv', index = False)


# ### Extracting IT Resumes from a Kaggle Dataset

# In[117]:


os.chdir(directory_path + "/../Input_Data/Resume_Data_Kaggle/Resume")


# In[118]:


ITR_df = pd.read_csv('Resume.csv')


# In[119]:


ITR_df.info()


# In[120]:


ITR_df['Category'].value_counts()


# In[121]:


Infotech_df = ITR_df[ITR_df["Category"] == "INFORMATION-TECHNOLOGY"]


# In[122]:


Infotech_df.info()


# In[125]:


os.chdir(directory_path + "/../Input_Data/")


# In[127]:


Infotech_df.to_csv('Kaggle_InfoTech_resumes_120.csv')

