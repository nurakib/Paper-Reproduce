import transformers
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "../language_model/bert-base-uncased"
MODEL_PATH = "model.bin"
CIRCA_DATASET = "../datasets/circa.tsv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lower_case=True
    )

LABEL_CODES = {
    'Yes': 0, 
    'No': 1, 
    'Yes, subject to some conditions': 2,
    'In the middle, neither yes nor no': 3 
}