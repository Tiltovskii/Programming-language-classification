from src.utils import *
import fasttext

print('Download train and test!')
x_train, x_test, y_train, y_test = download_train_test()
# x_train['code'] = x_train['code'].map(lambda x: x.rstrip())
# x_test['code'] = x_test['code'].map(lambda x: x.rstrip())

with open('data/final_data/ft_train.txt', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(x_train['code'], y_train['language']):
        f.writelines(f'__label__{each_label} {each_text}\n')

with open('data/final_data/ft_test.txt', 'w', encoding="utf-8") as f:
    for each_text, each_label in zip(x_test['code'], y_test['language']):
        f.writelines(f'__label__{each_label} {each_text}\n')

model = fasttext.train_supervised('data/ft_train.txt', epoch=100, wordNgrams=3,
                                  dim=512, loss='hs', thread=8, ws=7, lr=0.1)
model.save_model("model/ft_model.bin")


def print_results(sample_size, precision, recall):
    precision = round(precision, 2)
    recall = round(recall, 2)
    print(f'{sample_size=}')
    print(f'{precision=}')
    print(f'{recall=}')


# Применяем функцию
print_results(*model.test('data/ft_test.txt'))
