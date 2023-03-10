#!/usr/bin/env python
# coding: utf-8

# ## Importing the python libraries

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

from msedge.selenium_tools import Edge,EdgeOptions
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException


# ## Setting up the local Path for MS Edge Driver 

# In[ ]:


msedge_path = "C:\WebDriver\msedgedriver.exe"


# ### Total Number of Pages to scrap

# In[ ]:


TotalPC = 11


# ## Reading the list of Verticals from a file

# In[ ]:


Vertical_df = pd.read_csv('IT_Verticals.csv')


# In[ ]:


vert_list = Vertical_df['Vertical'].tolist()


# In[ ]:


len (vert_list)


# ### Building the list of URLs

# #### Unless there is a change in URL, we can use this list to save some time during each iteration of web scraping
# 
# #url_list = ['https://www.foundit.in/srp/results?query=Web%20Development', 'https://www.foundit.in/srp/results?query=Networking&searchId=eedca282-c380-4ffa-8dbd-2d7aa9017612', 'https://www.foundit.in/srp/results?query=Hardware%20ASIC%20Design', 'https://www.foundit.in/srp/results?query=Devops', 'https://www.foundit.in/srp/results?query=Cloud%20Computing', 'https://www.foundit.in/srp/results?query=Data%20Science', 'https://www.foundit.in/srp/results?query=Project%20Management', 'https://www.foundit.in/srp/results?query=Embedded%20Systems', 'https://www.foundit.in/srp/results?query=IT%20Support', 'https://www.foundit.in/srp/results?query=Information%20Security']

# In[ ]:


def get_current_url(url, job_title):
    driver = webdriver.Edge(msedge_path)
    driver.get(url)
    time.sleep(3)
    driver.find_element_by_xpath('//*[@id="SE_home_autocomplete"]').send_keys(job_title)
    time.sleep(3)
    #driver.find_element_by_xpath('//*[@id="SE_home_autocomplete_location"]').send_keys(location)
    #time.sleep(2)
    driver.find_element_by_xpath('//*[@type="submit"]').submit()
    current_url = driver.current_url
    driver.close()
    return current_url 


# In[ ]:


url_list = []

for i in range(len(vert_list)):
    current_url= get_current_url('https://www.foundit.in',vert_list[i])
    url_list.append(current_url)
print(url_list)


# In[ ]:


# Init the jobs_list as empty list
jobs_list = [] 


# In[ ]:


## This routine is used to,
## open the page for the given URL (input param)
## apply filter for "years of experience" (input param)
## apply filter for "recent postings in last 30 days"
## Click at each of the listed job cards
## Get text of each Job posting description (result of click)
## Append to the Jobs list.

def scrape_job_details(url, exp):
    
    options = EdgeOptions()
    options.use_chromium= True;
    options.add_argument("disable-infobars");
    options.add_argument("disable-popup-blocking");
    options.add_argument("disable-web-security");
    options.add_argument("inprivate");
    driver = Edge(executable_path="C:\WebDriver\msedgedriver.exe", options=options)
    
    driver.get(url)
    time.sleep(3)

    original_window = driver.current_window_handle
    
    element = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "top-filter-section")))
    
    driver.find_element_by_xpath('//ul[@class="filter-section"]//child::li[2]').click()
    time.sleep(2)
    if (exp <=5) :
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="2~5"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="1~2"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="0~1"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//button[@class="apply"]').click();
    elif (exp >5 and exp <=10):
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="5~7"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="7~10"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//button[@class="apply"]').click();
    elif (exp >10):
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="10~15"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="15~*"]').click();
        driver.find_element_by_xpath('//div[@class="filter-overlay"]//button[@class="apply"]').click();

    time.sleep(3)
    driver.find_element_by_xpath('//ul[@class="filter-section"]//child::li[8]').click()
    time.sleep(2)
    
    ### Filter for those entries posted in last 30 days
    driver.find_element_by_xpath('//div[@class="filter-overlay"]//input[@value="30"]').click();
    driver.find_element_by_xpath('//div[@class="filter-overlay"]//button[@class="apply"]').click();
    time.sleep(5)
    
    pc = 0
    while (pc < TotalPC):
        # Store the list of Job cards found in the page
        job_elems = driver.find_elements_by_xpath('//*[@class="srpResultCardContainer"]')

        for job in job_elems:
            try : 
                job.click()
            except ElementClickInterceptedException:
                if (driver.current_window_handle != original_window) :
                    driver.switch_to.window(original_window)
                    time.sleep(3)
                    job.click()
             
            except TimeoutException:
                continue;
            except StaleElementReferenceException:
                continue;
                    
                
            time.sleep(3)
            driver.find_elements_by_xpath('//*[@class="dbJdSection"]')
            #content = BeautifulSoup(driver.page_source, 'html5lib')
            content = BeautifulSoup(driver.page_source, 'html.parser')
            
          
            for post in content.select('.dbJdSection'):
                try:
                    data = {
                        "Job_Title":post.select('.jdTitle')[0].get_text().strip(),
                        "Company_Name":post.select('.jdCompanyName')[0].get_text().strip(),
                        "Location":post.select('.details')[0].get_text().strip(),
                        "Job_Description":post.select('.jobDescInfo')[0].get_text(strip=True, separator=",").strip()  
                    }
                except IndexError:
                    continue          
            jobs_list.append(data)
        
        try : 
            driver.find_element_by_xpath('//div[@class="pagination"]//div[@class="arrow arrow-right"]').click();
        except NoSuchElementException: 
            print("Last Page in Search Result")
            break;
        time.sleep(4)
        pc = pc + 1
        print("Total of ", TotalPC-1, "pages. Completed Page No :", pc)
            
    driver.close()


# In[ ]:


# Each Run in the loop is for a specific vertical.
# In each run, job postings for different levels of experience will be gathered
# and the dataframe is written to a csv file having pre-defined name format

for i in range(0,len(vert_list)):
     
    jobs_list = []
    start_time = time.time()
    scrape_job_details(url_list[i],5) 
    end_time = time.time()
    print(vert_list[i], url_list[i], "Exp : 0 to 5, Time Taken: ", end_time - start_time)
    jobs_dframe_exp_1_5 = pd.DataFrame(jobs_list)
    jobs_dframe_exp_1_5['Vertical'] = ''
    jobs_dframe_exp_1_5['Skills_List'] = ''
    jobs_dframe_exp_1_5['Years_of_Exp'] = ''
    file_name = vert_list[i].replace(" ","") + '_job_profile_exp1_5_foundit.xlsx'
    jobs_dframe_exp_1_5.to_excel(file_name, index=False)
    
    time.sleep(30)
    
    jobs_list = []
    start_time = time.time()
    scrape_job_details(url_list[i],10) 
    end_time = time.time()
    print(vert_list[i], "Exp : 5 to 10, Time Taken: ", end_time - start_time)
    jobs_dframe_exp_5_10 = pd.DataFrame(jobs_list)
    jobs_dframe_exp_5_10['Vertical'] = ''
    jobs_dframe_exp_5_10['Skills_List'] = ''
    jobs_dframe_exp_5_10['Years_of_Exp'] = ''
    file_name = vert_list[i].replace(" ","") + '_job_profile_exp6_10_foundit.xlsx'
    jobs_dframe_exp_5_10.to_excel(file_name, index=False)

    time.sleep(30)
    
    
    jobs_list = []
    start_time = time.time()
    scrape_job_details(url_list[i],11) 
    end_time = time.time()
    print(vert_list[i], "Exp : 10 plus, Time Taken: ", end_time - start_time)
    jobs_dframe_exp_10plus = pd.DataFrame(jobs_list)
    jobs_dframe_exp_10plus['Vertical'] = ''
    jobs_dframe_exp_10plus['Skills_List'] = ''
    jobs_dframe_exp_10plus['Years_of_Exp'] = ''
    file_name = vert_list[i].replace(" ","") + '_job_profile_exp_10plus_foundit.xlsx'   
    jobs_dframe_exp_10plus.to_excel(file_name, index=False)
    
    time.sleep(30)
    

