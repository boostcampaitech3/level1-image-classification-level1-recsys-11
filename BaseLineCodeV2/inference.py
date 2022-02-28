import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import dataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_model_select(saved_model, num_classes, device, which_model):
    model_cls = getattr(import_module("model"), which_model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

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
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


def inference_by_single_models(which_model, data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # which_model should be 'mask' or 'genderAge'
    model_class = None
    if which_model == 'mask':
        num_classes = 3
        # model_class = 'MaskSplitByProfileDatasetForAlbumOnlyMask'
    elif which_model == 'genderAge':
        num_classes = 6
        # model_class = 'MaskSplitByProfileDatasetForAlbumOnlyGenderAge'
    else:
        assert 'no match model type'

    model = load_model(model_dir, num_classes, device).to(device)
    model.eval() 

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

    print("Calculating inference " + which_model +  " inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output_' + which_model + '.csv')
    info.to_csv(save_path, index=False)
    print('Inference ' + which_model + ' complete saved as ' + save_path)

def ensemble_row(mask, genderAge):
    mask_str = str(mask)
    genderAge_str = str(genderAge)

    mask_genderAge_str = mask_str + genderAge_str

    mask_genderAge_dict = {
        '00' : 0,
        '01' : 1,
        '02' : 2,
        '03' : 3,
        '04' : 4,
        '05' : 5,
        '10' : 6,
        '11' : 7,
        '12' : 8,
        '13' : 9,
        '14' : 10,
        '15' : 11,
        '20' : 12,
        '21' : 13,
        '22' : 14,
        '23' : 15,
        '24' : 16,
        '25' : 17,
    }

    return mask_genderAge_dict[mask_genderAge_str]

def inference_ensemble(data_dir, model_dir, output_dir, args):

    mask_model_dir = input('mask 모델 저장 디렉토리 입력 >> ')
    genderAge_model_dir = input('genderAge 모델 저장 디렉토리 입력 >> ')

    inference_by_single_models('mask', data_dir, mask_model_dir, output_dir, args)
    inference_by_single_models('genderAge', data_dir, genderAge_model_dir, output_dir, args)

    df_mask = pd.read_csv(os.path.join(output_dir, f'output_' + 'mask' + '.csv'))
    df_genderAge = pd.read_csv(os.path.join(output_dir, f'output_' + 'genderAge' + '.csv'))

    df = pd.concat([df_mask, df_genderAge], axis=1, ignore_index=True)
    df = df.rename(columns={0:'ImageID', 1:'mask_ans', 2:'ImageID2', 3:'genderAge_ans'})
    df = df.drop('ImageID2', axis=1)
    df['ans'] = df.apply(lambda x: ensemble_row(x['mask_ans'], x['genderAge_ans']), axis=1)
    df = df.drop(['mask_ans', 'genderAge_ans'], axis=1)

    save_path = os.path.join(output_dir, f'output_' + 'ensemble' + '.csv')
    df.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[96, 128], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--dataset', type=str, default='TestDataset', help='TestDataset with data augmentation  (default: TestDataset)')
    parser.add_argument('--isEnsemble', type=str, default='False', help='Inference tpye True / False  (default: False)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.isEnsemble == 'False':
        inference(data_dir, model_dir, output_dir, args)
    elif args.isEnsemble == 'True':
        inference_ensemble(data_dir, model_dir, output_dir, args)
    else:
        assert 'check isEnsemble option (True/False  default:False)'

