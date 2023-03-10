# Import the necessary libraries
import pandas as pd

class skills_recommender:
  
  # Constructor of the class
  def __init__(self):
    # Read the data from an excel file and store it in a pandas dataframe
    self.df = pd.read_excel('./job_market_skills.xlsx')
    
    # Create a dictionary to map each vertical with its name
    self.vertical_mapping = {
        "vertical_0":"Embedded Development",
        "vertical_1":"Cloud Computing",
        "vertical_2":"Project Management",
        "vertical_3":"DevOps",
        "vertical_4":"Hardware Asic",
        "vertical_5":"IT Support",
        "vertical_6":"Networking",
        "vertical_7":"Web Development",
        "vertical_8":"Information Security"
    }

  # Method to get the matching vertical skills
  def get_matching_vertical_skills(self, skill_list):
    # Create an empty pandas dataframe to store the matching vertical skills
    return_df = pd.DataFrame(columns=['vertical','skill','score'])
    # Initialize a row index variable
    row_index=0
    # Loop through the skill_list
    for skill in skill_list:
      # Remove leading/trailing whitespaces from the skill
      skill.strip()

      # If the skill is not empty, then check if it matches with any of the skills in the dataframe
      if skill != '':
        for index, row in self.df.iterrows():
          # Get the list of skills in the current row and loop through them
          v_skills = row['skill'].split(' ')
          for _skill in v_skills:
            # Check if the skill matches with the current skill in the row
            if _skill.lower().strip() == skill.lower().strip():
              # If the skill matches, then add the corresponding vertical, skill, and score to the return dataframe
              return_df.loc[row_index] = [row['vertical'],row['skill'],row['score']]
              row_index = row_index+1
              #values_list.append([row['vertical'],row['skill'],row['score']])  
    # Return the dataframe containing the matching vertical skills
    return return_df

  # Method to get the top 10 skills for a given vertical
  def get_top_10_skills(self,vertical,org_skill_list):
    # Create an empty list to store the recommended skills
    skill_list = []
    # Initialize a skill count variable
    skill_count = 0
    # Get the rows from the dataframe corresponding to the given vertical and sort them by score in descending order
    temp_df = self.df[self.df['vertical'] == vertical]
    temp_df = temp_df.sort_values(['score'], ascending=[False])

    # Loop through the rows in the dataframe
    for index, row in temp_df.iterrows():
      # Get the current skill from the row
      new_skill = row['skill']
      # Check if the current skill is not already present in the org_skill_list and add it to the skill_list
      if new_skill not in org_skill_list:
        skill_list.append(new_skill)
        skill_count = skill_count+1
      # If the skill_count reaches 10, then break the loop
      if skill_count == 10:
        break
    # Return the recommended skill_list
    return skill_list

  # Method to get the name of a vertical given its code
  def get_vertical_name(self, vertical):
    return self.vertical_mapping[vertical]
        
  # Method to suggest skills based on a given list of skills
  def suggest_skills(self, org_skill_list):
    # Clean skill list if needed
    skill_list = []
    for _skill in org_skill_list:
        # If there are newline characters in the skill, replace them with commas
        if "\n" in _skill:
            _skill = _skill.replace("\n", ",")
            # Add each resulting skill to the skill_list
            skill_list = skill_list + _skill.split(',')
        else:
            skill_list.append(_skill)

    # Print the cleaned skill list for debugging purposes
    print("checking for skill list = ", skill_list)

    # Get a DataFrame with matching skills for each vertical
    df = self.get_matching_vertical_skills(skill_list)

    # Get the count of how many times each vertical appears in the DataFrame
    vertical_counts = df['vertical'].value_counts()

    # Get the unique list of verticals in the DataFrame
    verticals = df['vertical'].unique()

    # Create an empty dictionary to store scores for each vertical
    scores = {}

    # Calculate a score for each vertical based on the sum of the scores of the matching skills
    for vertical in verticals:
        # Initialize the vertical score to 0
        vertical_score = 0
        # Get a DataFrame with only the rows corresponding to the current vertical
        temp_df = df[df['vertical'] == vertical]
        # Calculate the sum of the scores for the matching skills
        vertical_score = temp_df['score'].sum()
        # Store the vertical score in the scores dictionary
        scores[vertical] = vertical_score

    # Sort the scores dictionary in descending order by value
    _scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    # Create a string that lists the original skills
    my_string = ','.join(skill_list)
    text = "Your current skills are: " + my_string
    text = text + "\n\n"

    # Define a dictionary to map index to recommended domain text
    rec_dict = {
        1: "Highest Recommended Domain: ",
        2: "\nSecond Best Recommended Domain: ",
        3: "\nAlternate Recommended Domain: ",
        4: "\nLess Preferable Recommended Domain: ",
        5: "\nVery Less Preferable Domain, But Is An Option: "
    }

    # Print the sorted scores for debugging purposes
    print(_scores)

    # Initialize the recommended domain index to 0
    rec_index = 0

    # Loop over the recommended domains and print them along with the top ten missing skills for each domain
    for key in _scores:
        # Increment the recommended domain index
        rec_index = rec_index + 1
        # If we have already printed the top five recommended domains, exit the loop
        if rec_index == 6:
            break
        # Add the recommended domain text to the output string
        text = text + rec_dict[rec_index] + self.get_vertical_name(key) + "\n"
        # Get the top ten skills missing from the original skill list for the current domain
        rec_list = self.get_top_10_skills(key, skill_list)
        i = 1
        # Print each missing skill along with its rank
        for _skill in rec_list:
            text = text + "   " + str(i) + ". " + _skill + "\n"
            i = i + 1

    # Return the output string
    return text


def print_db(self):
    # Print the DataFrame containing skill information
    print(self.df)

   
