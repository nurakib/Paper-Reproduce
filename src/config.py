import torch 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BERT_PATH = 'bert-base-uncased'
