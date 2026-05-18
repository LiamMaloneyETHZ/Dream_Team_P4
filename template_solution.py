# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import os
# We disable low-level log outputs by default to keep the terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that
# are not required

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16  # Set the batch size according to both training performance and available memory
NUM_EPOCHS = 5  # Set the number of epochs

train_val = pd.read_csv("train.csv")
test_val = pd.read_csv("test_no_score.csv")

# SentimentDataset
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, titles, sentences, labels=None):
        self.titles = titles
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        title = str(self.titles[index])
        sentence = str(self.sentences[index])

        text = title + " " + sentence

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        return item


train_dataset = SentimentDataset(
    train_val["title"].tolist(),
    train_val["sentence"].tolist(),
    train_val["score"].tolist()
)

test_dataset = SentimentDataset(
    test_val["title"].tolist(),
    test_val["sentence"].tolist()
)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
			  num_workers=0,         # If you want to utilize multi-processing, set this to the number of your available cores!
			  pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
			 num_workers=0,          # If you want to utilize multi-processing, set this to the number of your available cores!
			 pin_memory=True)

# Additional code if needed

# SentimentClassifier
class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.backbone.config.hidden_size
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 classes: negative (0) / positive (1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


model = SentimentClassifier().to(DEVICE)

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

best_loss = float("inf")
best_state = None

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss_sum = 0.0
    n_batches = 0

    for batch in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(output, batch["labels"])
        loss.backward()
        optimizer.step()

        epoch_loss_sum += loss.item()
        n_batches += 1

    mean_loss = epoch_loss_sum / n_batches
    marker = ""
    if mean_loss < best_loss:
        best_loss = mean_loss
        best_state = model.state_dict().copy()
        marker = " (best)"
    print(f"Epoch {epoch} mean loss: {mean_loss:.6f}{marker}")

    scheduler.step()

if best_state is not None:
    model.load_state_dict(best_state)

# Evaluation loop
model.eval()
with torch.no_grad():
    results = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        output = model(batch["input_ids"], batch["attention_mask"])
        preds = torch.argmax(output, dim=1)
        results.append(preds.cpu().numpy())

    with open("result.txt", "w") as f:
        for val in np.concatenate(results):
            f.write(f"{val}\n")

import winsound
winsound.Beep(1000, 1000)