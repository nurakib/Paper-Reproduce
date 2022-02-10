from operator import le
import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import os
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import parse_args
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    args = parse_args()

    df_test = pd.read_csv(args.test_data)

    print(len(df_test))

    # if hasattr(args, 'sample_percentage'): 
    #     df_test = df_test[:int(len(df_test) * args.sample_percentage / 100)]

    print(len(df_test))

    BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained(
                args.encoder_model, 
                do_lower_case=True
                )

    test_dataset = dataset.CIRCADataset(
        sentence_1 = df_test.sentence_1.values, 
        sentence_2 = df_test.sentence_2.values, 
        target = df_test.target.values,
        tokenizer = BERT_TOKENIZER,
        max_len = args.max_length
    )
    
    print(test_dataset[1])

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.train_batch_size, num_workers=2
    )

    print(f'Loaded: {len(test_dataset)} test samples')
    
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased(dropout=args.dropout, n_class=args.n_class)
    model.load_state_dict(torch.load(args.trained_model))
    model.to(device)

    _ , y_pred, y_test = engine.eval_fn(test_data_loader, model, device)
    df_test['y_pred'] = y_pred
    pred_test = df_test[['sentence_1', 'sentence_2', 'target', 'y_pred']]
    pred_test.to_csv(f'{args.result_dir}results_{args.experiment_name}.csv', index = False)
    print('Accuracy::', metrics.accuracy_score(y_test, y_pred))
    print('Precision::', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall::', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F_score::', metrics.f1_score(y_test, y_pred, average='weighted'))
    print('classification_report::')
    print(metrics.classification_report(y_test, y_pred))

    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{args.result_dir}report_{args.experiment_name}.csv', index = False)


if __name__ == "__main__":
    run()