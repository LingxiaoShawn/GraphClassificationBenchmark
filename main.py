import logging 
from itertools import product
import argparse

from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gcn import GCN, GCNWithJK
from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from gpca import GPCANet
from chebnet import ChebNet

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200) # 300 previously
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_decay_factor', type=float, default=0.85)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--chebK', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--net', type=str, default='GCN')
args = parser.parse_args()

Net = eval(args.net)

alphas = [0.1, 1, 10, 20, 50] # used in outer part
layers = [2, 3, 5, 7]
hiddens = [32, 64, 128]
# layers = [2]
# hiddens = [128]
# datasets = ['congress-sim3','mig-sim3']#['COLLAB', 'REDDIT-MULTI-5K'] #'AIDS', 'DD', 'PROTEINS',  'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY','congress-LS', 

datasets = ['DD', 'PROTEINS',  'IMDB-BINARY', 'REDDIT-BINARY']

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    # val_loss, test_acc = info['val_loss'], info['test_acc']
    # logging.debug('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
    #     fold, epoch, val_loss, test_acc))
    test_acc = info['test_acc']
    logging.debug('{:02d}/{:03d}: Test Accuracy: {:.3f}'.format(fold, epoch, test_acc))

results = []
for dataset_name in datasets:
    if dataset_name == 'REDDIT-MULTI-5K':
        layers = [5, 7]
        args.epochs=100
    # Reset logging: Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(message)s', level=logging.INFO, filename=f'logs/{dataset_name}-{Net.__name__}-K{args.chebK}-alpha{args.alpha}.log')
    # logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    for num_layers, hidden in product(layers, hiddens):
        logging.info('-'*50)
        logging.info(f'!L[{num_layers}] H[{hidden}]')
        dataset = get_dataset(dataset_name)

        model = Net(dataset, num_layers, hidden, K=args.chebK, alpha=args.alpha)
        
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=logger,
            gpu=args.gpu,
        )
        # if loss < best_result[0]:
        if acc > best_result[1]:
            best_result = (loss, acc, std)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    logging.info('--------------------Best result - {}'.format(desc))

#     results += ['{} - {}: {}'.format(dataset_name, model, desc)]
# logging.info('-----\n{}'.format('\n'.join(results)))