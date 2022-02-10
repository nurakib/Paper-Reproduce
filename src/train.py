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

from utils import parse_args
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')


def run():    
    args = parse_args()
        
    df_train = pd.read_csv(args.train_data)
    df_valid = pd.read_csv(args.valid_data)
    
    print(len(df_train), len(df_valid))

    if hasattr(args, 'sample_percentage'): 
        df_train = df_train[:int(len(df_train) * args.sample_percentage / 100)]
        df_valid = df_valid[:int(len(df_valid) * args.sample_percentage / 100)]

    print(len(df_train), len(df_valid))


    BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained(
                args.encoder_model, 
                do_lower_case=True
                )

    train_dataset = dataset.CIRCADataset(
        sentence_1 = df_train.sentence_1.values, 
        sentence_2 = df_train.sentence_2.values, 
        target = df_train.target.values,
        tokenizer = BERT_TOKENIZER,
        max_len = args.max_length
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, num_workers=2
    )

    valid_dataset = dataset.CIRCADataset(
        sentence_1 = df_valid.sentence_1.values, 
        sentence_2 = df_valid.sentence_2.values, 
        target = df_valid.target.values,
        tokenizer = BERT_TOKENIZER,
        max_len = args.max_length
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, num_workers=2
    )

    print(f'Loaded: {len(train_dataset)} training samples and {len(valid_dataset)} validation samples')
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased(dropout=args.dropout, n_class=args.n_class)
    if args.trained_model is not None: 
        print(f'Loaded model from: {args.trained_model}')
        model.load_state_dict(torch.load(args.trained_model))
    
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

    num_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)
    optimizer = AdamW(optimizer_parameters, lr = args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = num_train_steps
    )

    best_accuracy = 0
    training_stats = []
    for epoch in range(args.epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        print('Training...')
        avg_train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("")
        print("Running Validation...")
        avg_valid_loss, outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        accuracy = metrics.accuracy_score(targets, outputs)
        print("Average validation loss: {0:.2f}".format(avg_valid_loss))
        print("")
        print("Accuracy: {0:.2f}".format(accuracy))

        # Record all statistics from this epoch.
        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_valid_loss,
            'Validation Accuracy': accuracy
        })

        if accuracy > best_accuracy:
            MODEL_PATH = f"{args.save_model_dir}/model_{args.experiment_name}.bin"
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy
    print("======== Training Summary ========")
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    


if __name__ == "__main__":
    run()

