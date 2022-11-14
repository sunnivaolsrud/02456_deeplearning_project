
#%%%
import numpy as np
import pandas as pd
# import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
# from datasets import load_dataset
import datasets 

#%%

df = pd.read_csv(r'C:\\Users\\Tore\\02456_deeplearning_project\\src\\data\\twitter_data\\labels_tweets_dropna.csv')

vals = np.minimum(np.sum(df.to_numpy()[:,1:-1],1),1)
dataset = datasets.DatasetDict(
    {'train':datasets.Dataset.from_dict({'label':vals[:20000],'text':df['Tweet'][:20000]}),
     'test':datasets.Dataset.from_dict({'label':vals[20000:],'text':df['Tweet'][20000:]})
     })


#%%
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#%%
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# %% Define Train
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
) 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
#%%

trainer.train()
# %%
