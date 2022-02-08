import config
import torch


class CIRCADataset:
    def __init__(self, sentence_1, sentence_2, target):
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, item):
        sentence_1 = str(self.sentence_1[item])
        sentence_1 = " ".join(sentence_1.split())

        sentence_2 = str(self.sentence_2[item])
        sentence_2 = " ".join(sentence_2.split())

        inputs = self.tokenizer.encode_plus(
            sentence_1,
            sentence_2,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True 
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.long)
        }
