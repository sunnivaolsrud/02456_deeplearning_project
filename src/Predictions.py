
#%% Load data for model
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from datasets import load_dataset
from datasets import load_from_disk
import numpy as np

##### Load data for unbalanced dataset
d_a = pd.read_csv("data/twitter_data/new_tweet_dataset_test.csv")
d_a = d_a.rename(columns={'tweet_text':'Tweet'})
d_a['billed'] = d_a['Tweet'].apply(lambda x: 1 if 'http' in x else 0)
d_a_ub = d_a[d_a['billed']==0]
d_a_ub[["Tweet","label"]].to_csv("data/twitter_data/new_tweet_dataset_test.csv",index=False)

data_files = {"test": "data/twitter_data/new_tweet_dataset_test.csv"}
dataset = load_dataset("csv", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["Tweet"], padding="max_length", truncation=True)

tokenized_unbalanced = dataset.map(tokenize_function, batched=True)

#%%#### Load data (tokenizer) for balanced dataset (only for new prediction, old predictions are saved)
tokenized_balanced = load_from_disk("data/tokenized_data")



# Make new predictions by loading pretrained model and tokenizer 
    # Load model for predicting
model = AutoModelForSequenceClassification.from_pretrained(
             "bert_models/trained_bert_3", num_labels=2)

#### Prediction unbalanced
trainer1 = Trainer(model)
testset_unblanced = tokenized_unbalanced["test"]
predictions_unb, labels_unb, _, = trainer1.predict(testset_unblanced)
np.save('data/PredictedTestData/PredictionTestData_ub',predictions_unb)
np.save('data/PredictedTestData/Labels_ub',labels_unb)

#### Prediction balanced
trainer2 = Trainer(model)
testset_blanced   = tokenized_balanced["test"]
predictions_b, labels_b, _, = trainer2.predict(testset_blanced)
np.save('data/PredictedTestData/PredictionTestData_b',predictions_b)
np.save('data/PredictedTestData/Labels_b',labels_b)


