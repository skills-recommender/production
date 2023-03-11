BERTSkillsTopicModelling.py

This python file has code to extract skills from job postings.

Input files:
============
  1. Seed list of skills per verticals
      from production/data/Vertical_Skills_merged.csv

  2. Merged job postings from different web sites across different verticals
       from production/data/Merged_Data/

Output files:
=============
   1. Generates skill database and store it in production/data/job_market_skills.xlsx

   2. Stores generated model as binary file in production/data/GuidedBertopicModel

