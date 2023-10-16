from pathlib import Path
from tqdm import tqdm
from utils import *
from tokenizers import ByteLevelBPETokenizer


df = pd.read_parquet('data/dataWithoutComments.parquet')
x_train, x_test, y_train, y_test = download_train_test(df)

with open('data/ft_train.txt', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(x_train['code'], y_train['language']):
        f.writelines(f'{each_text}\n')

with open('data/ft_test.txt', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(x_test['code'], y_test['language']):
        f.writelines(f'{each_text}\n')


paths = [str(x) for x in Path('data/').glob('*.txt')]


# initialize
tokenizer = ByteLevelBPETokenizer()
# and train
tokenizer.train(files=paths, vocab_size=250000, min_frequency=2,
                special_tokens=['<|endoftext|>', '<s>', '<pad>', '</s>', '<unk>', '<mask>'], show)

tokenizer.save_model('model/tokenizer/without comments')
