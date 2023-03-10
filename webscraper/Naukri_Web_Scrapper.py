#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd


# ## Setting up the local Path for MS Edge Driver 

# In[ ]:


msedge_path = "C:\WebDriver\msedgedriver.exe"


# ### Total Number of Pages to scrap for each vertical

# In[ ]:


TotalPC = 35


# ## Reading the list of Verticals from a file

# In[ ]:


Vertical_df = pd.read_csv('IT_Verticals.csv')


# In[ ]:


vert_list = Vertical_df['Vertical'].tolist()


# ### Preparing a Dataframe to store the Job Postings data

# In[ ]:


df = pd.DataFrame(columns=['Vertical','Job_Title','Company_Name','Years_of_Exp','Location','Job_Description','Skills_List'])


# In[ ]:


url_list = []
url_list_page = []

for item in vert_list:
    item = item.lower()
    item1 = "https://www.naukri.com/" + item.replace(" ","-") + "-jobs"
    item2 = item.replace(" ","%20")
    url_list.append(item1)
    url_list_page.append(item2)


# ### Scrap the Job Description details from a URL - for each entry in the given page

# In[ ]:


def ExtractFromJobUrl(url, vertical):
    global df
    #lookup the URL
    driver = webdriver.Edge(r"msedgedriver.exe")
    driver.get(url)
    time.sleep(5)
    content = BeautifulSoup(driver.page_source, 'html5lib')
    
    for post in content.select('.leftSec'):        
        try:
            data = {
                    "Vertical":vertical,
                    "Job_Title":post.select('.jd-header-title')[0].get_text().strip(),
                    "Years_of_Exp":post.select('.exp')[0].get_text().strip(),
                    "Location":post.select('.location')[0].get_text().strip(),
                    "Job_Description":post.select('.dang-inner-html')[0].get_text().strip()                    
            }
            comp = post.find('div', class_='jd-header-comp-name')
            name = comp.find('a', class_='pad-rt-8')
            data["Company_Name"] = name.text
            
            ul = post.find('div',class_='key-skill')
            lis = []
            for li in ul.findAll('span'):
                lis.append(li)
    
            skills = ''
            for i in range(len(lis)):
                if (i != 0):
                    skills = skills + ', '
        
                skills = skills + lis[i].text

            data["Skills_List"] = skills
                        
        except IndexError:
            continue          
        
        df = df.append(data, ignore_index=True)
    
    driver.close()
    return


# ### Scrap all entries in the given page

# In[ ]:


def extractPage(url, verticalName):

    #lookup the URL
    driver = webdriver.Edge(r"msedgedriver.exe")
    driver.get(url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'html5lib')
    driver.close()
    
    #Extract data
    results = soup.find(class_='list')
    if results is None:
        print("result empty")
        return
    
    job_elems = results.find_all('article',class_='jobTuple')
    
    for job_elem in job_elems:
        Title = job_elem.find('a',class_='title ellipsis')
        jobUrl = job_elem.find('a',class_='title ellipsis').get('href')
        ExtractFromJobUrl(jobUrl, verticalName)
    return


# ### Scrap Job Descriptions for each vertical in sequence

# In[ ]:


for i in range(0,len(vert_list)):
    
    # First Page for the given vertical    
    url = url_list[i] + "?k=" + url_list_page[i] + "&jobAge=30"
    print("PageURL:", url)
    driver = webdriver.Edge(r"msedgedriver.exe")
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'html5lib')
    driver.close()
    results = soup.find(class_='list')
    job_elems = results.find_all('article',class_='jobTuple')
    for job_elem in job_elems:
        Title = job_elem.find('a',class_='title ellipsis')
        jobUrl = job_elem.find('a',class_='title ellipsis').get('href')
        ExtractFromJobUrl(jobUrl,vert_list[i])
    
    # Additional Pages for the same vertical
    for page in range(2,TotalPC+1):
        url = url_list[i] + "-" + str(page) + "?k=" + url_list_page[i] + "&jobAge=30"
        extractPage(url, vert_list[i])

    # Store the job details extracted in dataframe as excel file
    file_name = vert_list[i].replace(" ","") + '_job_profile_naukri.xlsx'
    df.to_excel(file_name, index=False)
    print("wrote df to excel for vertical:", vert_list[i])
    
    # clear the frame for storing next vertical
    df = df[0:0]

