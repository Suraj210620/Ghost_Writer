#Import necessary libraries for pre-trained model and tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

#Step2.1: Load the cleaned data file "paraphrases_cleaned.csv"
cleaned_data = pd.read_csv("paraphrases_cleaned.csv")
cleaned_data = Dataset.from_pandas(cleaned_data) #Converts a pandas dataframe to Huggingface dataset, optimized to use with transformers library

#Step2.2: Preprocessing the data
#Set the fixed max_length for all sequences
MAX_LENGTH = 1024
def preprocess_text(samples):
    inputs = ["paraphrase:" + text for text in samples['question1']]
    targets = [text for text in samples['question2']]
    tokenized_inputs = tokenizer(inputs,truncation = True, max_length = MAX_LENGTH,padding = 'max_length')['input_ids']
    tokenized_targets = tokenizer(targets,truncation = True, max_length = MAX_LENGTH,padding = 'max_length')['input_ids']
    return {"input_ids":tokenized_inputs,"labels":tokenized_targets}
    #return {"input_ids": tokenizer(inputs, truncation=True,padding=True,max_length=max_length)['input_ids'], 'labels':tokenizer(targets, truncation=True, padding=True, max_length=max_length)['input_ids']} 

#Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-large") #--> 't5_large' is the pretrained tokenizer from transformers
#Applying the preprocessing function to the cleaned_data
cleaned_data = cleaned_data.map(preprocess_text, batched=True)

#Step2.3: Fine tune the model
#Load the pretrained model
model = T5ForConditionalGeneration.from_pretrained("t5-large") #Need torch library

#Defining the training arguments
train_arg = TrainingArguments(output_dir="./t5-paraphraser",
                              per_device_train_batch_size=2,
                              num_train_epochs = 3,
                              logging_dir = "./log",
                              save_steps = 1000,
                              save_total_limit = 2)

#Initializing the trainer
trainer = Trainer(model = model, args = train_arg, train_dataset = cleaned_data) #Requires accelerate = 0.26.0 library
#Train the model
trainer.train()

#Saving the fine tuned model
model.save_pretrained("./t5_paraphraser")
tokenizer.save_pretrained("./t5_paraphraser")