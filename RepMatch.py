
# Import and setup

from IPython.display import clear_output

# !pip install datasets transformers torchmetrics evaluate
# !pip install --upgrade accelerate
# !pip install ipywidgets==7.7.1
# !pip install evaluate
# clear_output()



import torch
from torch import matmul, exp, log, abs
import torch.nn as nn
from torch.utils.data import DataLoader

from IPython.display import clear_output
from tqdm.notebook import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, BertModel, ElectraModel, set_seed
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from transformers import AdamW,get_scheduler
from transformers import AdamW,get_scheduler
from transformers import LlamaForSequenceClassification

from datasets import load_metric, load_dataset, Dataset

import random
import numpy as np
import evaluate
import math
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

import matplotlib.pyplot as plt

from numpy import linalg as LA

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting seed
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
set_seed(seed = SEED)

# language model hyper parameters
batch_size = 1
epochs = 1
lr = 1e-3
weight_decay = 0.1

BASE_MODEL_NAME = "bert-base-uncased"
# BASE_MODEL_NAME = 'google/electra-base-discriminator'
ALPHA=4.
RANK=1
target=['query', 'value']
NUM_LAYERS = 12

"""# Dataset"""

dataset_name = 'sst2'

if dataset_name == 'sst2':
    dataset = load_dataset('sst2')
    num_labels = 2
elif dataset_name == 'sst5':
    dataset = load_dataset('SetFit/sst5')
    num_labels = 5
elif dataset_name == 'imdb':
    dataset = load_dataset('imdb', verification_mode='no_checks')
    num_labels = 2
clear_output()

train_dataset = dataset["train"]

"""# Loading and tokenizing"""

model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels).to(device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Define LoRA Config
lora_config = LoraConfig(
 r = RANK,
 lora_alpha = ALPHA,
 target_modules = target,
 lora_dropout = 0,
 bias="none",
 task_type=TaskType.SEQ_CLS
)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.base_model.model.classifier.modules_to_save.default.requires_grad_(False)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print(train_dataset)
tokenized_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_valid = dataset['validation'].map(tokenize_function, batched=True)
# tokenized_test = dataset['test'].map(tokenize_function, batched=True)

tokenized_train.set_format("torch",columns=["input_ids", "attention_mask", "label"])
# tokenized_valid.set_format("torch",columns=["input_ids", "attention_mask", "label"])
# tokenized_test.set_format("torch",columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
N =len(tokenized_train)
"""# Training

"""

train_dataloader = DataLoader(
    tokenized_train, shuffle=False, batch_size=batch_size, collate_fn=data_collator
)
# valid_dataloader = DataLoader(
#     tokenized_valid, batch_size=batch_size, collate_fn=data_collator
# )
# test_dataloader = DataLoader(
#     tokenized_test, batch_size=batch_size, collate_fn=data_collator
# )

optim = torch.optim.AdamW(model.parameters(), lr=lr)

num_training_steps = epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optim,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )
print(num_training_steps)

f1_metric = load_metric("f1", average='macro', trust_remote_code=True)
acc_metric = load_metric("accuracy",trust_remote_code=True)

model = model.to(device)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optim.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')

"""## Test and train funcs

Load the final model, compute the deltaW for for layer l, and finally calculate svd.
"""

sst2_model_id = 'bert_sst2_rank4_value_classifier_frozen' # fine-tuned model
sst2_config = PeftConfig.from_pretrained(sst2_model_id)
sst2_model = AutoModelForSequenceClassification.from_pretrained(sst2_config.base_model_name_or_path, num_labels=num_labels)
final_model = PeftModel.from_pretrained(sst2_model, sst2_model_id)

final_model.eval()
clear_output()

q_target_matrices = [
    torch.matmul(
    final_model.base_model.electra.encoder.layer[l].attention.self.query.lora_B.default.weight,
    final_model.base_model.electra.encoder.layer[l].attention.self.query.lora_A.default.weight).to(device)
    for l in range(NUM_LAYERS)]

v_target_matrices = [
    torch.matmul(
    final_model.base_model.electra.encoder.layer[l].attention.self.value.lora_B.default.weight,
    final_model.base_model.electra.encoder.layer[l].attention.self.value.lora_A.default.weight).to(device)
    for l in range(NUM_LAYERS)]

q_target_svd = [torch.linalg.svd(q_target_matrices[l])[0] for l in tqdm(range(NUM_LAYERS))]
v_target_svd = [torch.linalg.svd(v_target_matrices[l])[0] for l in tqdm(range(NUM_LAYERS))]

q_distances = np.zeros((NUM_LAYERS, N))
v_distances = np.zeros((NUM_LAYERS, N))


checkpoint = torch.load('checkpoint.pth')

def new_train(train_model, optimizer, num_epochs, batch_size, train_dataloader, num_training_steps):
    index = 0
    progress_bar_train = tqdm(range(num_training_steps))
    # progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

    best_f1_metric = 0

    for epoch in range(num_epochs):

        train_model.train()

        # one step optimizer
        for batch in train_dataloader:
            # set seed
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            set_seed(seed = SEED)

            train_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_model.train()

            # idx = batch['idx']
            idx = index
            batch = {k: v.to(device) for k, v in batch.items()}
            # print(idx)
            # for i in range(1):
            optimizer.zero_grad()
            
            outputs = train_model(**batch)

            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            # lr_scheduler.step()

            progress_bar_train.update(1)

            with torch.inference_mode():
                
                # must have shape (NUM_LAYERS, d, d)
                v_train_matrices = [
                    torch.matmul(
                        train_model.base_model.electra.encoder.layer[l].attention.self.value.lora_B.default.weight,
                        train_model.base_model.electra.encoder.layer[l].attention.self.value.lora_A.default.weight)
                    for l in range(NUM_LAYERS)]
                
                q_train_matrices = [
                    torch.matmul(
                        train_model.base_model.electra.encoder.layer[l].attention.self.query.lora_B.default.weight,
                        train_model.base_model.electra.encoder.layer[l].attention.self.query.lora_A.default.weight)
                    for l in range(NUM_LAYERS)]


                #SVD
                q_train_svd = [torch.linalg.svd(q_train_matrices[l])[0] for l in range(NUM_LAYERS)]
                v_train_svd = [torch.linalg.svd(v_train_matrices[l])[0] for l in range(NUM_LAYERS)]

                
                for l in range(NUM_LAYERS):
                    q_grass = torch.zeros(RANK, RANK)
                    v_grass = torch.zeros(RANK, RANK)

                    for i in range(1, RANK+1):
                        for j in range(1, RANK+1):
                            q_grass[i-1][j-1] = torch.div(torch.pow(torch.norm(torch.matmul(q_target_svd[l][:, :i].T, q_train_svd[l][:, :j]), p='fro'), 2), min(i, j))
                            v_grass[i-1][j-1] = torch.div(torch.pow(torch.norm(torch.matmul(v_target_svd[l][:, :i].T, v_train_svd[l][:, :j]), p='fro'), 2), min(i, j))
                    q_distances[l][idx] = torch.max(q_grass)
                    v_distances[l][idx] = torch.max(v_grass)

                # If using rank 1
                # q_distances[:,idx] = np.array([torch.norm(torch.inner(q_target_svd[l][:,0], q_train_svd[l][:,0])).cpu().detach() for l in range(NUM_LAYERS)])
                # v_distances[:,idx] = np.array([torch.norm(torch.inner(v_target_svd[l][:,0], v_train_svd[l][:,0])).cpu().detach() for l in range(NUM_LAYERS)])
            # print(q_distances[:,idx])
            # print(v_distances[:,idx])

            print(index)
            index += 1

"""## Test and Train"""


new_train(model, optim,epochs, batch_size, train_dataloader, num_training_steps)

q_dist = np.array(q_distances)
v_dist = np.array(v_distances)

np.save('dist_electra_sst2_rank1_query_targetValueQuery_classifier_1.npy', q_dist)
np.save('dist_electra_sst2_rank1_value_targetValueQuery_classifier_1.npy', v_dist)