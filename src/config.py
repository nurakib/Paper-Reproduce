import torch 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BERT_PATH = 'bert-base-uncased'

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
