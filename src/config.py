import transformers
import torch
from datetime import datetime

date = datetime.now().strftime("%Y_%m_%d_%I_%M")

#datasets
CIRCA_TRAIN = "../datasets/circa/circa_train.csv"
CIRCA_VALID = "../datasets/circa/circa_valid.csv"
CIRCA_TEST = "../datasets/circa/circa_test.csv"

BOOLQ_TRAIN = "../datasets/boolq/boolq_train.csv"
BOOLQ_VAILD = "../datasets/boolq/boolq_valid.csv"

MNLI_TRAIN = "../datasets/mnli/mnli_train.csv"
MNLI_VALID = "../datasets/mnli/mnli_valid.csv"

NUMBER_OF_SAMPLES = 100000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
BERT_PATH = "bert-base-uncased"

MODEL_NAME = 'bert_mnli_short'
MODEL_PATH = f"../saved_models/model_{MODEL_NAME}.bin"
RESULT_PATH = "../results/"


TOKENIZER = transformers.BertTokenizer.from_pretrained(
                BERT_PATH, 
                do_lower_case=True
                )

LABEL_MAP = {
    'Yes': 0, 
    'In the middle, neither yes nor no': 1,
    'No': 2, 
    'Yes, subject to some conditions': 3
}

BOOLQ_LABEL_MAP = {
    'True': 1,
    'False': 0
}

NUMBER_OF_CLASSES = 3