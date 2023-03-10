#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import glob
directory_path = os.getcwd()


# In[ ]:


Vert_list = pd.read_csv('IT_Verticals.csv')


# In[ ]:


verticals = Vert_list['Vertical'].tolist()


# In[ ]:


Foundit_dict_title = {'job_title':'Job_Title', 'company':'Company_Name', 'location':'Location', 'job_desc':'Job_Description', 'vertical':'Vertical', 'Experience':'Years_of_Exp'}


# In[ ]:


Indeed_dict_title = {'job_title':'Job_Title', 'company':'Company_Name', 'location':'Location', 'job_desc':'Job_Description'}


# In[ ]:


cols =['Vertical', 'Job_Title', 'Company_Name', 'Years_of_Exp', 'Location',
       'Job_Description', 'Skills_List']


# In[ ]:


print (directory_path)

for vert in verticals:
    
    Vertical = vert.strip()
    vert = Vertical.replace(" ","")
    
    Foundit_df1 = pd.read_excel(f'./Foundit/{vert}_job_profile_foundit.xlsx')
    Naukri_df2 = pd.read_excel(f'./Naukri/{vert}_job_profile_naukri.xlsx')
    Indeed_df3 = pd.read_excel(f'./Indeed/{vert}_job_profile_indeed.xlsx')
    
    Foundit_df1.rename(columns = Foundit_dict_title, inplace=True)
    Foundit_df1['Skills_List'] = ''
    
    Indeed_df3.drop(['link'], axis=1, inplace = True)
    Indeed_df3.drop(['date'], axis=1, inplace = True)
    Indeed_df3.rename(columns = Indeed_dict_title, inplace=True)
    Indeed_df3['Skills_List'] = ''
    Indeed_df3['Years_of_Exp'] = ''
    Indeed_df3['Vertical'] = Vertical

    Foundit_ndf = Foundit_df1[cols]
    Naukri_ndf = Naukri_df2[cols]
    Indeed_ndf = Indeed_df3[cols]
    
    Foundit_ndf.to_excel(f'./Foundit_Dataset/{vert}_job_profile_foundit.xlsx',index= False )
    Naukri_ndf.to_excel(f'./Naukri_Dataset/{vert}_job_profile_naukri.xlsx',index= False)
    Indeed_ndf.to_excel(f'./Indeed_Dataset/{vert}_job_profile_indeed.xlsx', index= False)
    
    Merged_df = pd.concat([Foundit_ndf,Naukri_ndf,Indeed_ndf], axis="rows")
    Merged_df.to_excel(f'./Merged_Dataset/{vert}_job_profile.xlsx', index= False)


# In[ ]:


for vert in verticals:
    vert = vert.strip()
    vert = vert.replace(" ","")
    VDF = pd.read_excel(f'./Merged_Dataset/{vert}_job_profile.xlsx')
    print(vert, VDF.shape)
    

