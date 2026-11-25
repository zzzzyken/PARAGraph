#!/usr/bin/env python3
"""Standalone Swin test-set eval: load a best_model.pth, print CM + QWK/ACC/F1/SPE.
Usage: python evaluation/eval_swin_ckpt.py <ckpt.pth> <data_root> [split_seed]

<data_root> is a dataset folder in the fixed format (see README). Pass the split seed of the
training run when the folder has no test.csv, so the same test part is held out.
"""
import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import sys, numpy as np, torch
from torch.utils.data import DataLoader
from Models import SwinClassifier
from grading_dataloader import build_dataset, get_val_transform

ckpt_path = sys.argv[1]
data_root = sys.argv[2]
split_seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
a = ck['args']
args = a if isinstance(a, dict) else vars(a)
img = args.get('image_size', 1024)
swin_model = args.get('swin_model', 'swin_large_patch4_window7_224')
drop = args.get('drop_rate', 0.3)
num_classes = 5

device = 'cuda'
model = SwinClassifier(num_classes=num_classes, model_name=swin_model,
                       pretrained=False, drop_rate=drop, image_size=img).to(device)
model.load_state_dict(ck['model_state_dict'])
model.eval()

test_ds = build_dataset(data_root, 'test', get_val_transform(img), img,
                        use_3_stages=False, split_seed=split_seed)
loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

cm = np.zeros((num_classes, num_classes), dtype=np.int64)
with torch.no_grad():
    for batch in loader:
        x = batch['image'].to(device)
        y = batch['label'].numpy()
        with torch.autocast('cuda'):
            logits = model(x)['logits']
        pred = logits.argmax(1).cpu().numpy()
        for t, p in zip(y, pred):
            cm[t, p] += 1

def qwk(cm):
    N = cm.shape[0]
    w = np.fromfunction(lambda i, j: ((i-j)**2)/((N-1)**2), (N, N))
    act = cm.sum(1); pred = cm.sum(0); tot = cm.sum()
    E = np.outer(act, pred)/tot
    return 1 - (w*cm).sum()/(w*E).sum()

tot = cm.sum(); acc = np.trace(cm)/tot; f1s = []; spes = []
for i in range(num_classes):
    TP = cm[i, i]; FP = cm[:, i].sum()-TP; FN = cm[i, :].sum()-TP; TN = tot-TP-FP-FN
    f1s.append(2*TP/(2*TP+FP+FN) if (2*TP+FP+FN) > 0 else 0)
    spes.append(TN/(TN+FP) if (TN+FP) > 0 else 0)

print("Confusion Matrix:")
print(cm)
print(f"\nQWK={qwk(cm)*100:.2f}  ACC={acc*100:.2f}  F1={np.mean(f1s)*100:.2f}  SPE={np.mean(spes)*100:.2f}")
print("per-class F1:", [f"{f*100:.1f}" for f in f1s])
print("per-class SPE:", [f"{s*100:.1f}" for s in spes])
