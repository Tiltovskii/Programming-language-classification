import torch

from config import *
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaModel
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from utils import *

model = RobertaForSequenceClassification.from_pretrained(f"model/hf/", num_labels=len(LANGUAGES_TO_INT),
                                                         ignore_mismatched_sizes=True)
model.eval()
tokenizer = ByteLevelBPETokenizer("model/tokenizer/without comments/vocab.json",
                                  "model/tokenizer/without comments/merges.txt", )
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(length=512)

sentence = '''model = RobertaForSequenceClassification.from_pretrained(f"model/hf/",
              num_labels=len(LANGUAGES_TO_INT), ignore_mismatched_sizes=True)'''

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def main():
    enc_sent = tokenizer.encode(sentence)
    with torch.no_grad():
        logits = model(torch.tensor([enc_sent.ids]), workers=8)
    pred = logits[0].argmax()


if __name__ == '__main__':
    main()
