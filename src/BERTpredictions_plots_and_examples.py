
#%%%
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer
from datasets import load_from_disk
import evaluate

#%% Load Tokenizer and model for predicting
tokenized_imdb = load_from_disk("src/data/tokenized_data")
model = AutoModelForSequenceClassification.from_pretrained(
             "src/bert_models/trained_bert_3", num_labels=2)
trainer = Trainer(model)

# %% Prediction 
testset = tokenized_imdb["Validation"]
predictions, labels, _, = trainer.predict(testset)
metric = evaluate.load("accuracy")
print(metric.compute(predictions=np.argmax(predictions,-1), references=labels))

#%% Save/load predictions
# np.save('src/data/PredictedTestData/PredictionTestData',predictions)
# np.save('src/data/PredictedTestData/Labels',labels)
predictions = np.load('src/data/PredictedTestData/PredictionTestData.npy')
labels      = np.load('src/data/PredictedTestData/Labels.npy')

#%% examples
#accuarcy
print(np.sum(np.argmax(predictions,axis=-1)==labels)/200)
#examples
print(testset['Tweet'][:2], testset['Tweet'][175],testset['Tweet'][45])

#%% ROC curve
'We want a treshhold that gets most TP, a higher FP must then be the result'
'High TPR: Few sentenctes have been falesly classified as harassment'
exppred = np.exp(predictions)
predsoft = exppred/np.sum(exppred)
predsoft /= np.max(predsoft)
ratio = np.c_[[(predsoft[:,1]>i) for i in np.linspace(0.01,0.95,1000)]]

TP = np.sum((labels==ratio)[:,labels==1], 1)
TN = np.sum((labels==ratio)[:,labels==0], 1)
FP = np.sum((labels!=ratio)[:,labels==0], 1) 
FN = np.sum((labels!=ratio)[:,labels==1], 1)

TPR = TP/(TP+FN) # Recall/sensitivity
FPR = FP/(FP+TN)

plt.subplots(1, figsize=(10,10))
plt.title('ROC BERT')
plt.plot(FPR, TPR)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("50-epochs-roc")
#%% AUC
from sklearn import metrics
metrics.auc(FPR, TPR)

# %% Training loss visualization import data
import ast 
import matplotlib.pyplot as plt
import seaborn as sns

output = []
for i in ['Output_14988963']:
    with open(f'src/log/{i}.out') as f:
        output.append(np.array(f.readlines())[1:])
messagedict =  [list(map(ast.literal_eval,run)) for run in output]

mapp = lambda x: [[epoch[x] for epoch in run if x in epoch] for run in messagedict]
eval_loss=mapp('eval_loss');eval_accu=mapp('eval_accuracy');train_loss=mapp('loss') 

num_epochs = 10
train_loss_ = np.array(train_loss[0])[np.linspace(0,len(train_loss[0])-1,10).astype(int)]
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots()
sns.lineplot(x=np.arange(1,num_epochs+1), y=train_loss_, color='blue', label='Train');
sns.lineplot(x=np.arange(1, num_epochs + 1), y=eval_loss[0], color='orange', label='Valid');
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("test.png")