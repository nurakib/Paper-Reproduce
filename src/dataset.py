import config
import torch


class CIRCADataset:
    def __init__(self, question, answer, target):
        self.question = question
        self.answer = answer
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        question = str(self.question[item])
        question = " ".join(question.split())

        answer = str(self.answer[item])
        answer = " ".join(answer.split())

        inputs = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float)
        }