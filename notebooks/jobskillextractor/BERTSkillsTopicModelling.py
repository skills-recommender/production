#!/usr/bin/env python
# coding: utf-8

#
# Import required libraries
#
import re
import pandas as pd
from textblob import TextBlob
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.representation import MaximalMarginalRelevance

import nltk
nltk.download('stopwords')
nltk.download('brown')

#
# Import Seed list of skill sets from all verticals
#
directory_path = os.getcwd() 
os.chdir(directory_path + "/data/")
df_skill = pd.read_csv("Vertical_Skills_merged.csv")

#
# Create a list of list for seed words
#
columns = list(df_skill)
skill_lol = []
for col in columns:
    skills_list  = df_skill[col].tolist()
    skills_split = skills_list[0].split(",")
    skill_lol.append(skills_split)


#
# create combined data frame with job postings from all websites
# across different verticals
#
dfs = []
verticals = ["CloudComputing_job_profile",
             "DataScience_job_profile",
             "DevOps_job_profile",
             "EmbeddedSystems_job_profile",
             "HardwareASICdesign_job_profile",
             "ITSupport_job_profile",
             "InformationSecurity_job_profile",
             "Networking_job_profile",
             "ProjectManagement_job_profile",
             "WebDevelopment_job_profile"]

for vertical in verticals:
  temp_df = pd.read_excel(f'/Merged_Data/{vertical}.xlsx')
  dfs.append(temp_df)
  del temp_df

df = pd.concat(dfs, sort=False, ignore_index=True)
del dfs


punc_list = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',
    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']

#
# function to remove puntuations
#
def remove_puncutation(text,punc_list=punc_list):
  for punc  in punc_list:
    text = text.replace(punc, '')
  return text

#
# function to remove extra spaces
#
def remove_extra_space(text):
  return " ".join(text.split())

#
# function to extrace nouns from a given sentence
#
def get_noun_phrases(sentence):
  blob = TextBlob(sentence)
  return [w for (w, pos) in TextBlob(sentence).pos_tags if pos[0] == 'N']

from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#
# Clean up job description
#
df['clean_job'] = df['job_desc'].str.lower()
df['clean_job'] = df['clean_job'].apply(remove_puncutation)
df['clean_job'] = df['clean_job'].apply(remove_extra_space)
df['clean_job'] = df['clean_job'].str.replace('\d+', '')

#
# Remove stop words and extract nouns
#
stop_words = stopwords.words('english')
df['clean_job'] = df['clean_job'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df['clean_job'] = df['clean_job'].apply(lambda x: ' '.join([word for word in x.split() if word.isalpha()]))
df['clean_job'] = df['clean_job'].apply(get_noun_phrases)

#
#convert list to string
#
df['clean_job'] = [','.join(map(str.strip, l)) for l in df['clean_job']]

#
# combine skills with cleaned job description and create
# a list of preprocessed documents
#
df['clean_job'] = df['clean_job'] + df['skills']
preprocessed_documents = df.clean_job.tolist()

#
# Create BERTopic Model
#
from bertopic import BERTopic

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
representation_model = MaximalMarginalRelevance(diversity=0.5)

#
# Following parameters are used for bertopic model
#  - seed list: for guided bertopic modelling
#  - trigram vectorizer
#  - Bert Sentence Transformer as embedding model
#  - MMR as representation model to ensure diversity of topic words
#
topic_model = BERTopic(n_gram_range=(1,3),
                       seed_topic_list=skill_lol,
                       embedding_model=sentence_model,
                       representation_model=representation_model)

#
# Feed in preprocessed documents to the model and get results
#
topics, probs = topic_model.fit_transform(preprocessed_documents)

#
# Get topics generated
#
topic_model.get_topic_info()

#
# Visualize words per topic for first few topics
#
topic_model.visualize_barchart()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#
# function to print wordcloud of a given topic
#
def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Show wordcloud output for a topic
create_wordcloud(topic_model, topic=5)

#
# Print top 10 topic words from all topics
#
topic_model.get_topics()

#
# Below code scans through topic words from multiple topics generated
# by the model and maps them to one of 10 verticals. Then, forms
# skill database with verticals, skills and tf-idf scores fields.
#
skill_df = pd.DataFrame(columns=['vertical','skill','score'])

total_results = len(topic_model.get_topic_info())
topic_list = []

for i in range(1,total_results-1):
  topic_words = []
  for j in range(10):
    topic_words.append(topic_model.get_topic(i)[j])
  topic_list.append(topic_words)
  
#Find vertical based on most match
topic_class = 0
topic_index = 0
topic_score = 0
verticals = [[] for i in range(len(skill_lol))]
row_index = 0.#check each topic one by one
skipped_words = []

for current_topic_words in topic_list:
  #match with all seed to get final index
  final_vertical_index = 0
  final_vertical_score = 0  
  index = 0
  for seeds in skill_lol:
    current_vertical_score = 0

    for word,score in current_topic_words:
      if word in seeds:
        current_vertical_score = current_vertical_score + 1
    
      if current_vertical_score > final_vertical_score:
        final_vertical_score = current_vertical_score
        final_vertical_index = index
    index = index+1
  #skip if score is less than 4
  if final_vertical_score < 4:
    for _word,score in current_topic_words:
      skipped_words.append(_word)
    continue  
    
  for _word,score in current_topic_words:
    verticals[final_vertical_index].append(_word)
    vertical_name = 'vertical_'+str(final_vertical_index)
    skill_df.loc[row_index] = [vertical_name,_word,score]
    row_index = row_index+1


#
# Display number of skill keywords per vertical
#
skill_df['vertical'].value_counts()

#
# check difference between seed and verticals for new skills identified by the model
#
for i in range(len(verticals)):
  print(list(set(verticals[i]) - set(skill_lol[i])))

#
# Store skill db generated as a excel file. This will be used
# as input db in job skill recommendation system to the end user.
#
skill_df.to_excel("job_market_skills.xlsx")

#
# Save the trained model for later use
#
topic_model.save("GuidedBertopicModel")
