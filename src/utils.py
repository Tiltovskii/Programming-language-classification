from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from glob import glob
from config import LANGUAGES
from transformers.data.metrics import simple_accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score


def visiulize_roc_auc(model: RandomForestClassifier, x_train: np.array, y_train: np.array,
                      x_test: np.array, y_test: np.array, classes: list) -> None:
    r_forest_cm = ConfusionMatrix(model,
                                  classes=classes)

    r_forest_cm.fit(x_train, y_train)
    r_forest_cm.score(x_test, y_test)
    r_forest_cm.show()


def download_train_test(data: pd.DataFrame = None, test_size: int = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not glob('data/final_data/x_train.parquet') or data is not None:
        xy_train, xy_test = train_test_split(data, test_size=test_size,
                                             random_state=random_state)
        x_train, x_test, y_train, y_test = (
            xy_train[['code']],
            xy_test[['code']],
            xy_train[['language']],
            xy_test[['language']]
        )
        for name, file in zip(['x_train', 'x_test', 'y_train', 'y_test'],
                              [x_train, x_test, y_train, y_test]):
            file.to_parquet(f'data/final_data/{name}.parquet')
    else:
        x_train, x_test, y_train, y_test = (
            pd.read_parquet('data/final_data/x_train.parquet'),
            pd.read_parquet('data/final_data/x_test.parquet'),
            pd.read_parquet('data/final_data/y_train.parquet'),
            pd.read_parquet('data/final_data/y_test.parquet')
        )
    return x_train, x_test, y_train, y_test


def save_model(vectorizer, model) -> None:
    with open('model/finalized_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, model), f)


def removeSpecialComments(code, specialChars, shell=False):
    lines = code.splitlines()
    temp = []
    for line in lines:
        if shell:
            tempLine = line.strip()
        else:
            tempLine = line
        if tempLine == '':
            continue
        if tempLine[0] in specialChars:
            temp.append(line)
    for line in temp:
        lines.remove(line)
    code = "\n".join(lines)
    return code


def RemoveComments(code, language):
    identifier = LANGUAGES[language]
    if len(identifier) % 2 == 0:
        for k in range(0, len(identifier), 2):
            start = identifier[k]
            end = identifier[k + 1]
            temp = []
            commentIndex = []
            for i in range(len(code)):
                if code[i: i + len(start)] == start:
                    for j in range(i + len(start) + 1, len(code)):
                        if code[j: j + len(end)] == end and code[j - 1: j - 1 + len(start)] != start:
                            commentIndex.append((i, j + len(end)))
                            break
            temp[:0] = code
            for l, m in reversed(commentIndex):
                del temp[l:m + len(end)]
            code = ''.join(temp)
    if language == "FORTRAN":
        code = removeSpecialComments(code, ['*', 'C', 'c', 'd', 'D'])
    elif language == 'Shell':
        code = removeSpecialComments(code, ['#'], True)
    return code


def evaluate(model, eval_dataset) -> tuple[float, float]:
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = np.empty(0, dtype=np.int64)
    out_label_ids = np.empty(0, dtype=np.int64)

    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=128)

    for step, (input_ids, labels) in enumerate(tqdm(dataloader, desc="Eval")):
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"))
            loss = outputs[0]
            logits = outputs[1]
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
        preds = np.append(preds, logits.argmax(dim=1).detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        # del input_ids
        # del labels
        # del outputs
        # torch.cuda.empty_cache()
    eval_loss = eval_loss / nb_eval_steps
    acc = simple_accuracy(preds, out_label_ids)
    f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")

    print("=== Eval: loss ===", eval_loss)
    print("=== Eval: acc. ===", acc)
    print("=== Eval: f1 ===", f1)

    return eval_loss, acc


def tokenize_batch(batch: np.array, tokenizer) -> list:
    encoding = tokenizer.encode_batch(batch)
    en_train_data = [e.ids for e in encoding]
    return en_train_data
