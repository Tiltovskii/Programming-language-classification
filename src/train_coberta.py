from dataset import CodeDataset
from config import *
from tqdm import trange, tqdm
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaModel
from utils import *
from torch.utils.data import WeightedRandomSampler
import wandb
from collections import Counter
import os

run = wandb.init(
    # set the wandb project where this run will be logged
    project="Tgml",

    # track hyperparameters and run metadata
    config={
        "learning_rate": LR,
        "architecture": "codeBerta",
        "epochs": EPOCH,
        'optimizer': 'Adam'
    }
)

trainData = np.load('data/TOKENIZEDtrainData.npy')
validationData = np.load('data/TOKENIZEDvalidationData.npy')
trainLabels = pd.read_parquet('data/y_train.parquet').to_numpy()
validationLabels = pd.read_parquet('data/y_test.parquet').to_numpy()

train_dataset = CodeDataset(trainData, trainLabels, LANGUAGES_TO_INT)
eval_dataset = CodeDataset(validationData, validationLabels, LANGUAGES_TO_INT)

model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_PRETRAINED, num_labels=len(LANGUAGES_TO_INT),
                                                         ignore_mismatched_sizes=True)

count = Counter(trainLabels.T[0])
count = {k: 100/v for k, v in count.items()}
squarer = lambda x: count[x]
vfunc = np.vectorize(squarer)
weights = vfunc(trainLabels.T[0])

sampler = WeightedRandomSampler(weights, 150000, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

model.to("cuda")
model.train()
for param in model.roberta.parameters():
    param.requires_grad = False

print(f"num params:", model.num_parameters())
print(f"num trainable params:", model.num_parameters(only_trainable=True))

trainLA = []
validationLA = []
best_loss = np.inf
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
for e in trange(0, EPOCH, desc="Epoch"):
    train_loss = 0.0
    nb_train_steps = 0
    preds = np.empty(0, dtype=np.int64)
    out_label_ids = np.empty(0, dtype=np.int64)
    for step, (input_ids, labels) in enumerate(tqdm(train_dataloader, desc="Iteration")):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"))
        loss = outputs[0]
        loss.backward()
        logits = outputs[1]
        train_loss += loss.mean().item()
        nb_train_steps += 1
        preds = np.append(preds, logits.argmax(dim=1).detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        optimizer.step()
        # del input_ids
        # del labels
        # del outputs
        # torch.cuda.empty_cache()
    train_loss = train_loss / nb_train_steps
    acc = simple_accuracy(preds, out_label_ids)
    f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
    print("=== Train: loss ===", train_loss)
    print("=== Train: acc. ===", acc)
    print("=== Train: f1 ===", f1)

    wandb.log({"train_loss": train_loss, "train_acc": acc})

    val = evaluate(model, eval_dataset)
    if val[0] < best_loss:
        best_loss = val[0]
        model.save_pretrained(f"model/hf/")

    trainLA += [[train_loss, acc]]
    validationLA += [val]
    wandb.log({"val_loss": val[0], "val_acc": val[1]})
    model.train()
wandb.finish()
