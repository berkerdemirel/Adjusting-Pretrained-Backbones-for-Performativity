import sys

import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

sys.path.append('../../..')


class DistilBertClassifier(DistilBertForSequenceClassification):
    """
    Adapted from https://github.com/p-lambda/wilds
    """

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


def get_transform(arch, max_token_length):
    """
    Adapted from https://github.com/p-lambda/wilds
    """
    if arch == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(arch)
    else:
        raise ValueError("Model: {arch} not recognized".format(arch))

    def transform(text):
        tokens = tokenizer(text, padding='max_length', truncation=True,
                           max_length=max_token_length, return_tensors='pt')
        if arch == 'bert_base_uncased':
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif arch == 'distilbert-base-uncased':
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform