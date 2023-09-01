import os
import shutil
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold

from model import STGraphormer
from data import transforms
from data.preprocessing import make_data
from data.feeder import Feeder


def fix_seed(seed: int) -> None:
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'fixed seeds: {seed}')


def main(cfg):
    # save config file
    base_path = f'{cfg.path.save}'
    os.makedirs(base_path, exist_ok=True)
    shutil.copyfile(args.config_file, f'{base_path}/config.yaml')

    # seeds
    seeds = cfg.seeds

    # preprocess data
    X, y = make_data(
        path=cfg.path.load, 
        if_save=False,
        verbose=True)

    for seed in seeds:
        fix_seed(seed)
        save_path = f'{base_path}/seed{seed}'
        os.makedirs(save_path, exist_ok=True)

        # leave one out loop
        kf = StratifiedKFold(n_splits=5)
        for idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            trail_save_path = f'{save_path}/Trial{idx+1:03}'
            os.makedirs(trail_save_path, exist_ok=True)
            # split data
            print(f'Trail #{idx+1}')
            X_train = X[train_idx].copy()
            y_train = y[train_idx].copy()

            X_test = X[test_idx].copy()
            y_test = y[test_idx].copy()

            # dataset
            train_set = Feeder(
                X_train, 
                y_train,         
                transforms=transforms.Compose([
                        # transforms.Shear(r=cfg.dataset.shear_amplitude),
                        # # transforms.Gaus_noise(),
                        transforms.RandomFlip(flip_pair=cfg.dataset.flip_pairs),
                        # transforms.AddVelAcc(add_vel=cfg.dataset.add_vel, add_acc=cfg.dataset.add_acc)
                        # transforms.OnlyVelAcc(add_vel=cfg.dataset.add_vel, add_acc=cfg.dataset.add_acc)
                        # transforms.AddBoneChannel(inward=cfg.dataset.inward)
                    ]),
            )
            test_set = Feeder(
                X_test, 
                y_test, 
                transforms=transforms.Compose([
                        # transforms.AddVelAcc(add_vel=cfg.dataset.add_vel, add_acc=cfg.dataset.add_acc)
                        # transforms.OnlyVelAcc(add_vel=cfg.dataset.add_vel, add_acc=cfg.dataset.add_acc)
                        # transforms.AddBoneChannel(inward=cfg.dataset.inward)
                    ]),
            )

            # dataloader    
            train_loader = DataLoader(
                dataset=train_set, 
                batch_size=cfg.dataset.batch_size*len(cfg.device.id), 
                num_workers=os.cpu_count(), 
                pin_memory=True,
                shuffle=True,
            )
            test_loader = DataLoader(
                dataset=test_set, 
                batch_size=cfg.dataset.batch_size*len(cfg.device.id), 
                num_workers=os.cpu_count(), 
                pin_memory=True,
                shuffle=False,
            )

            # model
            model = eval(cfg.model.name)(**cfg.model.args)

            if cfg.device.cuda:
                model.cuda(cfg.device.id[0])
                if cfg.device.parallel:
                    model = torch.nn.DataParallel(
                        model, 
                        device_ids=cfg.device.id)

            # optimizer
            optimizer = eval(cfg.optim.name)(
                model.parameters(), 
                **cfg.optim.args)

            if cfg.skd.name:
                scheduler = eval(cfg.skd.name)(
                    optimizer, 
                    **cfg.skd.args)

            # criterion
            criterion = eval(cfg.criterion.name)()

            # Creates once at the beginning of training
            scaler = torch.cuda.amp.GradScaler()
            for epoch in range(1, cfg.num_epoch + 1):
                model.train()
                correct = 0
                sum_loss = 0
                
                for batch_idx, (data, label) in enumerate(train_loader):
                    optimizer.zero_grad()
                    if cfg.device.cuda:
                        data = data.cuda(cfg.device.id[0], non_blocking=True)
                        label = label.cuda(cfg.device.id[0], non_blocking=True)

                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, label)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    sum_loss += loss.item()
                    _, predict = torch.max(output.data, 1)
                    correct += (predict == label).sum().item()
                if cfg.skd.name:
                    scheduler.step()
                if epoch % 10 == 0:
                    print(f'# Epoch: {epoch:03}/{cfg.num_epoch:03} | Loss: {sum_loss/len(train_loader.dataset):.3f} | Accuracy: {100.*correct/len(train_loader.dataset):.1f}')
                    torch.save(model.to('cpu').state_dict(), f'{trail_save_path}/model_{epoch:03}.pth')
                    if cfg.device.cuda:
                        model.cuda(cfg.device.id[0])
            model.eval()

            correct = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(test_loader):
                    if cfg.device.cuda:
                        data = data.cuda(cfg.device.id[0], non_blocking=True)
                        label = label.cuda(cfg.device.id[0], non_blocking=True)

                    output = model(data)
                    _, predict = torch.max(output.data, 1)
                    correct += (predict == label).sum().item()
                    # result
                    with open(f'{save_path}/result.csv', 'a') as f:
                        [f.write(f'{label[i]},{predict[i]},{output[i, 0].item():.3f},{output[i, 1].item():.3f}\n') for i in range(len(label))]
            print('# Test Accuracy: {:.3f}[%]'.format(100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    import argparse
    import yaml
    from dotmap import DotMap

    arg_parser = argparse.ArgumentParser(description="TrianFlow testing.")
    arg_parser.add_argument('-c', '--config_file',
                            help='config file.')
    args = arg_parser.parse_args()

    # load yaml file
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg, _dynamic=False)   # dict to dot notation

    main(cfg)