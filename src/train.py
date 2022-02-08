from operator import le
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    df_train = pd.read_csv(config.MNLI_TRAIN)
    df_valid = pd.read_csv(config.MNLI_VALID)

    #for protype
    df_train = df_train[:config.NUMBER_OF_SAMPLES]
    df_valid = df_valid[:config.NUMBER_OF_SAMPLES]

    # print(df_train)

    train_dataset = dataset.CIRCADataset(
        sentence_1 = df_train.sentence_1.values, 
        sentence_2 = df_train.sentence_2.values, 
        target = df_train.target.values
    )
    
    # print(train_dataset[1])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=1
    )

    valid_dataset = dataset.CIRCADataset(
        sentence_1 = df_valid.sentence_1.values, 
        sentence_2 = df_valid.sentence_2.values, 
        target = df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    print(f'Loaded: {len(train_dataset)} training samples and {len(valid_dataset)} validation samples')
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:

            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()

