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

#Validation
#Import necessary libraries for validation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

#Step3.1:Load the Fine-tuned T5 model from preprocessing
t5_model = "./t5_paraphraser"
tokenizer = T5Tokenizer.from_pretrained(t5_model)
model = T5ForConditionalGeneration.from_pretrained(t5_model)

#Step3.2:Loads the Sentence-BERT[Bidirectional Encoder Representations from Transformers] Model
sbert_model = SentenceTransformer("all-MiniLM-L6-V2") #"all-MiniLM-L6-V2" is the pretrained sentence BERT Model

#Step3.3:Build the function for paraphrasing text
def paraphrase(input_text):
    inputs = tokenizer("paraphrase:"+input_text,return_tensors = 'pt')
    outputs = model.generate(inputs['input_ids'], max_length=200, num_beams = 8,length_penalty = 2.0, early_stopping = True)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_characters = True)
    return paraphrased_text

#Step3.4:Build the function to validate simantic similarity
def validate_simantic_similarity(original,overwritten):
    embeddings = sbert_model.encode([original,overwritten],convert_to_tensors=True)
    simantic_similarity = util.cos_sim(embeddings[0],embeddings[1])
    return simantic_similarity.item()

#Step3.5:Integrate both the functions into one
def paraphrase_similarity(input_text):
    paraphrased_text = paraphrase(input_text)
    similarity_score = validate_simantic_similarity(input_text,paraphrased_text)
    return paraphrased_text, similarity_score

#Usage
original_text = "Explain AI"
paraphrased_text,similarity_score = paraphrase_similarity(original_text)
print(f"Original:",{original_text})
print(f"Paraphrased_text:",{paraphrased_text})
print(f"Similarity_score:",{similarity_score})
