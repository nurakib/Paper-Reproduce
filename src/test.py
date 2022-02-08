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

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    df_test = pd.read_csv(config.CIRCA_TEST)

    #for protype
    df_test = df_test[0:100]

    # print(df_test)

    test_dataset = dataset.CIRCADataset(
        sentence_1 = df_test.sentence_1.values, 
        sentence_2 = df_test.sentence_2.values, 
        target = df_test.target.values
    )
    
    # print(test_dataset[1])

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2
    )

    device = torch.device(config.DEVICE)

    model = BERTBaseUncased()
    model.load_state_dict(torch.load(f'../saved_models/model_{config.MODEL_NAME}.bin'))
    model.to(device)

    y_pred, y_test = engine.eval_fn(test_data_loader, model, device)
    df_test['y_pred'] = y_pred
    pred_test = df_test[['sentence_1', 'sentence_2', 'target', 'y_pred']]
    pred_test.to_csv(f'{config.RESULT_PATH}results_{config.MODEL_NAME}.csv', index = False)
    print('Accuracy::', metrics.accuracy_score(y_test, y_pred))
    print('Precision::', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall::', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F_score::', metrics.f1_score(y_test, y_pred, average='weighted'))
    print('classification_report::')
    print(metrics.classification_report(y_test, y_pred, target_names=config.LABEL_MAP))

    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f'{config.RESULT_PATH}report_{config.MODEL_NAME}.csv', index = False)


if __name__ == "__main__":
    run()