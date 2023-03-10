#!/usr/bin/env python
# coding: utf-8

# 

# In[6]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys

HEADERS ={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}


# In[7]:


msedge_path = "C:\WebDriver\msedgedriver.exe"


# In[10]:


def get_current_url(url, job_title, location):
    #driver = webdriver.Chrome()
    driver = webdriver.Edge(msedge_path)
    driver.get(url)
    time.sleep(3)
    driver.find_element_by_xpath('//*[@id="text-input-what"]').send_keys(job_title)
    time.sleep(3)
    driver.find_element_by_xpath('//*[@id="text-input-where"]').send_keys(location)
    time.sleep(3)
    
    try:
        #driver.find_element_by_xpath('//*[@id="jobsearch"]/button').click()
        driver.find_element_by_xpath('//*[@type="submit"]').submit()   
    except:
        driver.find_element_by_xpath('//*[@id="whatWhereFormId"]/div[3]/button').click()
    
    current_url = driver.current_url

    return current_url 


# In[11]:


def scrape_job_details_direct(url):
    driver = webdriver.Edge(msedge_path)
    driver.get(url)
    time.sleep(3)

    jobs_list = []
    content = BeautifulSoup(driver.page_source, 'html5lib')
    print(content.prettify())
    
    
    #for post in content.select('.job_seen_beacon'):
    try:
        data = {
            "job_title":content.find('h1',{'class':'icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title'}).get_text().strip(),
            "job_desc":content.find('div',{'class':'jobsearch-jobDescriptionText'}).get_text().strip(),
            "company":content.find('div',{'class':'css-czdse3 eu4oa1w0'}).get_text().strip()
            
            #"company":content.select('.companyName')[0].get_text().strip(),
           # "rating":post.select('.ratingNumber')[0].get_text().strip(),
           # "location":content.select('.jobLocation')[0].get_text().strip(),
           # "date":content.select('.datePosted')[0].get_text().strip(),
           
        }
    except:
        return  

    print(data)
    jobs_list.append(data)
    dataframe = pd.DataFrame(jobs_list)
    return dataframe

#data_f = scrape_job_details_direct('https://in.indeed.com/viewjob?jk=d2bb89d763adeac8&fccid=fffc8238af387b00&vjs=3')


# In[12]:


def get_description(url):
    driver = webdriver.Edge(r"msedgedriver.exe")
    driver.get(url)
    time.sleep(3)

    content = BeautifulSoup(driver.page_source, 'html5lib')
    #print(content.prettify())
    driver.close()
    
    jd = None
    try:
        jd = content.find('div',{'class':'jobsearch-jobDescriptionText'}).get_text().strip()
    except IndexError:
        return jd
    return jd

#print(get_description('https://in.indeed.com/viewjob?jk=3aa72c94afbc18e3&fccid=9ab138c02aaca829&vjs=3'))


# In[19]:


def scrape_job_details(url):
    #resp = requests.get(url) #, headers=HEADERS)
    
    driver = webdriver.Edge(r"msedgedriver.exe")
    driver.get(url)
    time.sleep(3)
    
    content = BeautifulSoup(driver.page_source, 'html5lib')
    #print(content.prettify())
    driver.close()

    jobs_list = []
    
    for post in content.select('.job_seen_beacon'):
        try:
            links=post.find('a',{'class':'jcs-JobTitle'})['href']
            #split on "?" and take later part only
            parts = links.split("?")
            links = "https://in.indeed.com/viewjob?"+parts[1]
            
            data = {
                    "Job_Title":post.select('.jobTitle')[0].get_text().strip(),
                    "Company_Name":post.select('.companyName')[0].get_text().strip(),
                    "Location":post.select('.companyLocation')[0].get_text().strip(),
                    "Job_Description":"",
                    "link":links 
                }
        except:
            continue     
        
        jobs_list.append(data)
    #print(jobs_list)
    #-----------------------------------------------------------------------------------------------    
    ########################### Now we go through these direct links to get details ################
    #-----------------------------------------------------------------------------------------------
    
    
    for data in jobs_list:
        new_url = data["link"]
        desc = get_description(new_url)
        data['Job_Description'] = desc
        
    dataframe = pd.DataFrame(jobs_list)
    return dataframe


# In[ ]:


Vertical_df = pd.read_csv('IT_Verticals.csv')
vert_list = Vertical_df['Vertical'].tolist()
len (vert_list)


# In[21]:


maxId = 11
jump=10

for i in range(0,len(vert_list)):
    current_url = get_current_url('https://in.indeed.com/',vert_list[i],"India")
    url = current_url
    jobs_dframe = scrape_job_details(url)
    
    for page in range(10,maxId):
        url = current_url+"&start="+str(page)
        temp_f = scrape_job_details(url)

        jobs_dframe = pd.concat([jobs_dframe, temp_f], ignore_index=True, sort=False)
        page= page+jump
    
    jobs_dframe['Vertical'] = vert_list[i]
    jobs_dframe['Skills_List'] = ''
    jobs_dframe['Years_of_Exp'] = ''
    
    try:
        ## link, date - Not used as of now.
        jobs_dframe.drop(['link'], axis = 1, inplace=True)
    except KeyError:
        print("Key error encountered")
    
    file_name = vert_list[i].replace(" ","") + '_job_profile_indeed.xlsx'
    data_f.to_excel(file_name, index=False)


# In[ ]:




