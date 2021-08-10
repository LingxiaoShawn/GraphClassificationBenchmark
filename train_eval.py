import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
import logging

NUM_WORKERS = 0

def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None, gpu=0):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    # print(device)
    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        # train_dataset = dataset[train_idx]
        train_dataset = dataset[torch.cat([train_idx, val_idx])]
        test_dataset = dataset[test_idx]
        # val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=NUM_WORKERS)
        # val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=NUM_WORKERS)

        model.to(device).reset_parameters()

        if hasattr(model, 'init'):
            loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            data = next(iter(loader))
            model.init(data.to(device))
            data.to('cpu')

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader, device)
            logging.debug(f"Train Loss: {train_loss}")
            # val_losses.append(eval_loss(model, val_loader, device))
            accs.append(eval_acc(model, test_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                # 'val_loss': val_losses[-1],
                'test_acc': max(accs), #accs[-1],
            }
            

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    # loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    # loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    # loss, argmin = loss.min(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    acc, duration = tensor(accs).view(folds, epochs), tensor(durations)
    acc, _ = acc.max(dim=1)
    # loss_mean = loss.mean().item()
    loss_mean = 0
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    logging.info('!Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
                    format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        # print(loss.item())
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)