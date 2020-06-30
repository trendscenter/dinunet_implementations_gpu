from core.models import VBMNet
from classification import NiftiDataset, iteration, evaluation
from core.torchutils import NNDataLoader, initialize_weights
from torch.utils.data import ConcatDataset
from core.models import VBMNet
import torch
import torch.nn.functional as F
import os
import json
from core.measurements import Prf1a
import torch.nn as nn


def get_dataset(conf, fold, split_key=None):
    dataset = NiftiDataset(files_dir=f"test/input/local{s}/simulatorRun/{conf['data_dir']['value']}",
                           labels_dir=f"test/input/local{s}/simulatorRun/{conf['label_dir']['value']}",
                           mode='pooled')
    split_file = f"test/input/local{s}/simulatorRun/{conf['split_dir']['value']}/SPLIT_{fold}.json"
    split = json.loads(open(split_file).read())
    dataset.load_indices(split[split_key])
    return dataset


def eval(data_loader, model, device):
    score = Prf1a()
    for i, batch in enumerate(data_loader):
        inputs, labels = batch['inputs'].to(device).float(), batch['labels'].to(device).long()
        out = F.log_softmax(model(inputs), 1)
        _, preds = torch.max(out, 1)
        sc = Prf1a()
        sc.add(preds, labels)
        score.accumulate(sc)
    return score


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    R = 8
    LR = 0.001
    BZ = 16
    device = torch.device('cuda')
    epochs = 111
    os.makedirs('pooled_log', exist_ok=True)
    global_score = Prf1a(0)
    for fold in range(10):

        train, val, test = [], [], []
        for s, conf in enumerate(json.loads(open('test/inputspec.json').read())):
            train.append(get_dataset(conf, fold, 'train'))
            val.append(get_dataset(conf, fold, 'validation'))
            test.append(get_dataset(conf, fold, 'test'))

        train_dset = ConcatDataset(train)
        train_loader = NNDataLoader.new(dataset=train_dset, batch_size=BZ,
                                        pin_memory=True, shuffle=True, drop_last=True)

        val_dset = ConcatDataset(val)
        val_loader = NNDataLoader.new(dataset=val_dset, batch_size=BZ, pin_memory=True, shuffle=True)

        test_dset = ConcatDataset(test)
        test_loader = NNDataLoader.new(dataset=test_dset, batch_size=BZ, pin_memory=True, shuffle=True)

        print(f'fold {fold}:', len(train_dset), len(val_dset), len(test_dset))

        model = nn.DataParallel(VBMNet(1, 2, r=R))
        model = model.to(device)
        initialize_weights(model)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        best_score = 0.0
        for ep in range(epochs):
            for i, batch in enumerate(train_loader):
                inputs, labels = batch['inputs'].to(device).float(), batch['labels'].to(device).long()

                optim.zero_grad()
                out = F.log_softmax(model(inputs), 1)
                loss = F.nll_loss(out, labels)
                loss.backward()
                optim.step()

                _, preds = torch.max(out, 1)
                score = Prf1a()
                score.add(preds, labels)
                if i in list(range(11)) or i % 5 == 0:
                    print(f'Ep:{ep}/{epochs}, Itr:{i}/{len(train_loader)}, {round(loss.item(), 4)}, {score.prfa()}')
            val_score = eval(val_loader, model, device)
            if val_score.f1 > best_score:
                best_score = val_score.f1
                torch.save(model.state_dict(), f'pooled_log/best_{fold}.pt')
                print(f'##### *** BEST saved ***  {best_score}')
            else:
                print('###### Not Improved:', val_score.f1, best_score)

        model.load_state_dict(torch.load(f'pooled_log/best_{fold}.pt'))
        test_score = eval(test_loader, model, device)
        global_score.accumulate(test_score)
        with open(f'pooled_log/{fold}_prfa.txt', 'w') as wr:
            wr.write(test_score.prfa())
            wr.flush()

    with open(f'pooled_log/global_prfa.txt', 'w') as wr:
        wr.write(global_score.prfa())
        wr.flush()
