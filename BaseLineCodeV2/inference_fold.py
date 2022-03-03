import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import dataset
import albumentations as A
import ttach as tta

def load_models(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    models = []
    for i in range(5):
        model = model_cls(
            num_classes=num_classes
        )
        model_path = os.path.join(saved_model, args.state+f'_{i}.pth' ) # default : best.pth
        model.load_state_dict(torch.load(model_path, map_location=device))

        # args.tta = False
        if args.tta == True : 
            print("TTA will be applied !")
            my_tta_transforms = tta.Compose([
                    tta.HorizontalFlip()
                ])
            model = tta.ClassificationTTAWrapper(model, my_tta_transforms)
        else :
            print("No TTA !")
        
        models.append(model)
    
    return models

def load_model_select(saved_model, num_classes, device, which_model):
    model_cls = getattr(import_module("model"), which_model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, args.state+'.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    # models = load_models(model_dir, num_classes, device).to(device)
    models = [model.to(device) for model in load_models(model_dir, num_classes, device)]

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]


    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: TestDataset
    dataset = dataset_module(img_paths, args.resize)
    # dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            for model_idx, model in enumerate(models):
                if model_idx == 0:
                    pred = model(images)
                else :
                    pred += model(images)
            # pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[96, 128], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--dataset', type=str, default='TestDataset', help='TestDataset with data augmentation  (default: TestDataset)')
    parser.add_argument('--state', type=str, default='best', help='which state do you want to use. options are `best`, `last`, `best_f1` (default: best)')
    parser.add_argument('--tta', type=bool, default='False', help='Using TTA ; True / False  (default: False)')


    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    inference(data_dir, model_dir, output_dir, args)
