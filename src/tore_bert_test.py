
#%%
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk
from evaluate import load
#%% Load tokenized dataset and model
tokenized_data = load_from_disk("data/tokenized_data")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

#%% Define arguments
metric = load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
            output_dir="test_trainer", 
            evaluation_strategy="epoch",
            num_train_epochs=3)

trainer = Trainer(
    model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
)

#%% Train and save model
trainer.train()
trainer.save_model('bert_models/trained_bert_3')
