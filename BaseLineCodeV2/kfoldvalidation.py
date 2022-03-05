from train import *
import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from collections import Counter

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import mlflow
import mlflow.pytorch

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # -- experiment & save dir
    model_dir = os.path.join(model_dir, args.experiment)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    
    # -- mode ; train mode. If you wanna train All train data, change it to 'all'
    # -- data_loader
    n_splits = 5
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # kf = KFold(n_splits=n_splits, shuffle=True)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    mask_label = np.array(dataset.mask_labels)
    gender_label = np.array(dataset.gender_labels)
    age_label = np.array(dataset.age_labels)

    y_label = mask_label*100 + gender_label*10 + age_label
    print('-----------------------------------------------------------------------')
    # for fold, (train_ids, valid_ids) in enumerate(kf.split(dataset)):
    for fold, (train_ids, valid_ids) in enumerate(kf.split(dataset, y_label)):
        print(f'--------------------------------FOLD: {fold}--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_subsampler,
            num_workers=multiprocessing.cpu_count() // 2,
            # shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            dataset,
            batch_size=args.valid_batch_size,
            sampler=valid_subsampler,
            num_workers=multiprocessing.cpu_count() // 2,
            # shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        best_val_f1 = 0

        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0

            # Iterate over the DataLoader for training data
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                if args.model in 'Inception': # for inception v3
                    outs = outs.logits
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    # train_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro") ######################### add f1score
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch:2}/{args.epochs:2}]({idx + 1:4}/{len(train_loader):4}) || "
                        f"training loss {train_loss:6.4} || training accuracy {train_acc:6.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                    mlflow.log_metric("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    mlflow.log_metric("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                    

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                pred_list = [] # for calculating f1 score
                label_list = [] # for calculating f1 score

                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    pred_list.extend(preds.cpu().numpy()) # for calculating f1 score
                    label_list.extend(labels.cpu().numpy()) # for calculating f1 score

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / (len(dataset)/n_splits)
                val_f1 = f1_score(label_list, pred_list, average= 'macro')

                best_val_loss = min(best_val_loss, val_loss)
                # best_val_f1  = max(best_val_f1, val_f1)

                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.4%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_{fold}.pth")
                    mlflow.pytorch.log_model(model, 'bestModel')
                    best_val_acc = val_acc

                if val_f1 > best_val_f1:
                    print(f"New best model for val f1-macro : {val_f1:4.4%}! saving the best f1 model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best_f1_{fold}.pth")
                    mlflow.pytorch.log_model(model, 'bestModel')
                    best_val_f1 = val_f1


                torch.save(model.module.state_dict(), f"{save_dir}/last_{fold}.pth")
                print(
                    f"[Val] acc : {val_acc:4.4%}, loss: {val_loss:4.4}, f1 score ['macro'] : {val_f1:4.4}   || "
                    f"best acc : {best_val_acc:4.4%}, best loss: {best_val_loss:4.4}, best f1 score: {best_val_f1:4.4} ",   
                )
                print()

                # initialize the acc of each class
                class_total = Counter(range(18))
                class_total.subtract(class_total)
                class_corrects = class_total.copy()

                class_total.update(Counter(label_list))
                class_corrects.update(Counter(np.array(label_list)[np.array(label_list) == np.array(pred_list)]))

                for i in range(18):
                    print(f"| Class {i:2} ",end='')
                print("|")
                for i in range(18):
                    print(f"| {class_corrects[i]:^8} ",end='')
                print("|")
                for i in range(18):
                    print(f"| {class_total[i]:^8} ", end='')
                print("|")
                for i in range(18):
                    print(f"| {divide_except_zero(class_corrects[i], class_total[i]):^8.4} ",end='')
                print("|")


                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/f1score(macro)", val_f1, epoch)
                logger.add_figure("results", figure, epoch)

                mlflow.log_metric("Val/loss", val_loss, epoch)
                mlflow.log_metric("Val/accuracy", val_acc, epoch)
                mlflow.log_metric("Val/f1-macro", val_f1, epoch)

                mlflow.log_metric("best Val/loss", best_val_loss, epoch)
                mlflow.log_metric("best Val/accuracy", best_val_acc, epoch)
                mlflow.log_metric("best Val/f1-macro", best_val_f1, epoch)

                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    remote_server_uri ='http://101.101.210.160:30001'
    mlflow.set_tracking_uri(remote_server_uri)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=422, help='input batch size for validing (default: 422)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', type=str, default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--mode',type=str, default='split', help="choose the method of training using valid or not (default: split. If you want to train using all dataset, change it as 'all')")
    parser.add_argument('--user', type=str, default='unknown', help='set experiment username')

    # Container environment
    parser.add_argument('--experiment', type=str, default='general', help='set experiment name (default: general)')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))


    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    
    experiment_name_dict = {
        'general'  : "GENERAL", #default
        'mask'     : "Mask_model_experiment", # Mask Task 
        'genderAge': "enderAge_model_experiment" # genderAge Task
    }
    experiment_name = experiment_name_dict[args.experiment]
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = mlflow.tracking.MlflowClient()

    run = client.create_run(experiment.experiment_id)

    with mlflow.start_run(run_id=run.info.run_id):
        # mlflow.set_tag('mlflow.runName', run_name)
        mlflow.set_tag('mlflow.user', args.user)
        mlflow.log_params(args.__dict__)
        train(data_dir, model_dir, args)
