from utils import *
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

df = pd.read_parquet('data/preprocessed_data/dataWithoutComments.parquet')
df = df.dropna()
x_train, x_test, y_train, y_test = download_train_test(df)
x_train = x_train.values.flatten().tolist()
x_test = x_test.values.flatten().tolist()
del df

tokenizer = ByteLevelBPETokenizer("model/tokenizer/without comments/vocab.json",
                                  "model/tokenizer/without comments/merges.txt")
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(length=512)

res = []
for i in tqdm(range(len(x_train) // 10000)):
    res += tokenize_batch(x_train[10000 * i: 10000 * (i + 1)], tokenizer)
res += tokenize_batch(x_train[10000 * (i+1):], tokenizer)
np.save('data/final_data/TOKENIZEDtrainData.npy', res)

res = []
for i in tqdm(range(len(x_test) // 10000)):
    res += tokenize_batch(x_test[10000 * i: 10000 * (i + 1)], tokenizer)
res += tokenize_batch(x_test[10000 * (i+1):], tokenizer)
np.save('data/final_data/TOKENIZEDvalidationData.npy', res)
