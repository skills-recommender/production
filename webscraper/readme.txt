=========================================================
Dependencies :

MS Edge Driver needs to be installed in the system
and the path for the Driver should be known

=========================================================
Input Data :

Path for the MS Edge Driver need to updated in the script
msedge_path = "C:\WebDriver\msedgedriver.exe"

TotalPC = Value => Total number of Pages to scrap 
This count varies from site to site.

IT_Verticals.csv - To get the list of Verticals for which web scraping has to be done

=========================================================
Python Libraries :

pip install msedge-selenium-tools 
pip install html5lib
pip install requests
pip install beautifulsoup4
pip install msedge-selenium-tools selenium==3.141

=========================================================
How to execute :

Run Naukri script - Naukri_Web_Scrapper.py, generated files for each vertical will be stored in production/data/Naukri_Data/
Run Indeed script - Indeed_Web_Scraper.py, generated files for each vertical will be stored in production/data/Indeed_Data/
Run Foundit script - Foundit_Web_Scrapper.py, generated files for each vertical will be stored in production/data/Foundit_Data/

After completion of all the above three scripts,

Run Job_sites_Data_Merge_Per_Vertical.py to merge the data from all sites Vertical wise
Generated files will be stored in  production/data/Merged_Data/
=========================================================

