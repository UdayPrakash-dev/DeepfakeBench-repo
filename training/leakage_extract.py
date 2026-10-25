"""
Extract UCF branch features for the leakage / disentanglement analysis.

Run from the repo root (same way as test.py):

  # the released (authors') checkpoint
  python training/leakage_extract.py \
      --detector_path ./training/config/detector/ucf_smoke.yaml \
      --weights_path  ./training/weights/ucf_best.pth \
      --out ./leakage_feats/released.npz

  # control: same architecture at its ImageNet initialisation, NO UCF training
  python training/leakage_extract.py \
      --detector_path ./training/config/detector/ucf_smoke.yaml \
      --no_weights --out ./leakage_feats/control.npz

Saved arrays, for each split in {train, test}:
  <split>_content  (N,512)  global-avg-pooled output of encoder_c  (content branch)
  <split>_forgery  (N,512)  global-avg-pooled output of encoder_f  (before the spe/sha split)
  <split>_spe      (N,256)  specific feature  f_spe   (input of head_spe)
  <split>_sha      (N,256)  common feature    f_share (input of head_sha)
  <split>_prob     (N,)     model's P(fake) from the common head
  <split>_y_spe    (N,)     0=real 1=Deepfakes 2=Face2Face 3=FaceSwap 4=NeuralTextures

Frames: every `stride`-th frame of the 32 stored per video (default 8 -> 4 frames/video, spread
over the video), no augmentation.
"""
import os
import sys
import json
import argparse
import random

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # .../training
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


def build_config(detector_path):
    with open(detector_path, 'r') as f:
        cfg = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        cfg2 = yaml.safe_load(f)
    cfg2['label_dict'] = cfg['label_dict']        # keep UCF's 5-class labels (as train.py does)
    cfg.update(cfg2)
    cfg['frame_num'] = {'train': 32, 'test': 32}  # load all frames, subsample with --stride
    cfg['use_data_augmentation'] = False
    return cfg


def make_loader(cfg, split, stride, bs):
    c = dict(cfg)
    if split == 'train':
        c['train_dataset'] = ['FaceForensics++']
    else:
        c['test_dataset'] = 'FaceForensics++'
    ds = DeepfakeAbstractBaseDataset(c, mode=split)
    sub = Subset(ds, list(range(0, len(ds), stride)))
    return DataLoader(sub, batch_size=bs, shuffle=False,
                      num_workers=int(cfg['workers']), collate_fn=ds.collate_fn)


@torch.no_grad()
def extract(model, loader, device):
    out = {k: [] for k in ['content', 'forgery', 'spe', 'sha', 'prob', 'y_spe']}
    for batch in tqdm(loader):
        x = batch['image'].to(device, non_blocking=True)
        feats = model.features({'image': x})
        fz, ct = feats['forgery'], feats['content']
        f_spe, f_share = model.classifier(fz)
        _, spe_feat = model.head_spe(f_spe)        # = global-avg-pool(f_spe)
        logit_sha, sha_feat = model.head_sha(f_share)  # = global-avg-pool(f_share)
        out['content'].append(ct.mean(dim=(2, 3)).float().cpu().numpy())
        out['forgery'].append(fz.mean(dim=(2, 3)).float().cpu().numpy())
        out['spe'].append(spe_feat.float().cpu().numpy())
        out['sha'].append(sha_feat.float().cpu().numpy())
        out['prob'].append(torch.softmax(logit_sha, dim=1)[:, 1].float().cpu().numpy())
        out['y_spe'].append(batch['label'].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detector_path', required=True)
    p.add_argument('--weights_path', default=None)
    p.add_argument('--no_weights', action='store_true',
                   help='control run: ImageNet-initialised encoders, no UCF training')
    p.add_argument('--out', required=True)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--bs', type=int, default=16)
    args = p.parse_args()
    if args.no_weights == (args.weights_path is not None):
        sys.exit('give exactly one of --weights_path or --no_weights')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = build_config(args.detector_path)
    seed = cfg.get('manualSeed') or 1024
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = DETECTOR[cfg['model_name']](cfg).to(device)
    if not args.no_weights:
        ckpt = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('loaded', args.weights_path)
    else:
        print('CONTROL: no UCF weights (ImageNet-initialised encoders, random heads/adjust layers)')
    model.eval()

    save = {}
    for split in ['train', 'test']:
        loader = make_loader(cfg, split, args.stride, args.bs)
        print(f'[{split}] {len(loader.dataset)} images')
        for k, v in extract(model, loader, device).items():
            save[f'{split}_{k}'] = v

    # sanity check: the model's own AUC on the test split (released checkpoint should be ~0.95+)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(save['test_y_spe'] > 0, save['test_prob'])
    print(f'sanity: common-head AUC on test subset = {auc:.4f}'
          + ('  (control: expect ~0.5, heads are untrained)' if args.no_weights else ''))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save['meta'] = np.array(json.dumps({'weights': args.weights_path, 'stride': args.stride,
                                        'control': args.no_weights, 'sanity_auc': float(auc)}))
    np.savez_compressed(args.out, **save)
    print('saved', args.out)


if __name__ == '__main__':
    main()
