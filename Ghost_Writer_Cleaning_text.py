import pandas as pd #For Data Manupulation
import re #For Regular expressuions for text processing

#Step1.1: Collecting the data from local directory.
data = pd.read_csv("questions.csv")
#print(data)
#Step1.2: Filter and Prepare the data 
paraphrases = data[data['is_duplicate']==1][['question1','question2']]
#print(paraphrases)
#Step1.3: Cleaning the text data 
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]','',text)
    text = text.lower().strip()
    return text

#Applying the function "clean_text" to both the question columns
paraphrases['question1'] = paraphrases['question1'].apply(clean_text)
paraphrases['question2'] = paraphrases['question2'].apply(clean_text)

#Now saving the cleaned file which is eliminated special charecters and numerics
paraphrases.to_csv("paraphrases_cleaned.csv",index=False)