import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self, dropout, n_class):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(dropout)
        self.out = nn.Linear(768, n_class)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output