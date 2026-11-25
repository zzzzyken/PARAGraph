# PARAGraph

Official implementation of **PARAGraph: Pathology-Anatomy-Aware Hierarchical Graph for
Diabetic Retinopathy Grading**, accepted at **IEEE BIBM 2026**.

## What the model does

PARAGraph turns one fundus image into a small graph and grades DR (0-4) from it.

- **Structure from two frozen segmentation networks.** `OpticUNet` (ResNet-50) gives the
  optic disc (OD) mask and a fovea heat-map. `LesionUNet` (Swin-T) gives four lesion
  masks: microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and soft exudates (SE).
  Both run at 1024 px.
- **An anatomical coordinate system.** The OD centre is the origin and distances are
  measured in OD diameters. The fovea, taken as the soft-argmax of its heat-map, is the
  second reference point and fixes the reference direction.
- **A three-level graph.** Each detected lesion is a node. Lesions of one type connect to a
  category node. K-means clusters of lesions form region nodes (4 by default). A graph node
  anchored at the OD and a global node anchored at the fovea collect the result.
- **Visual features with an OD positional bias.** Node features come from a Swin-T encoder
  (512 px). Its tokens carry an OD-relative position code: the distance from the OD in OD
  diameters and the angle to the OD-fovea direction.
- **Confidence gating.** Every node has a gate `g = sigmoid(a * c + b)` driven by its
  segmentation confidence `c`. An edge is weighted by the product of its two gates, so a
  message counts only when both ends are reliable.
- **Two output branches.** Graph attention layers produce the anatomical logits. A global
  head on the same Swin encoder produces global logits. The prediction is their sum.

The full description is in the docstrings of [Models/paragraph.py](Models/paragraph.py).

## Repository layout

```
PARAGraph/
├── Models/                      PARAGraph, the segmentation networks, all baseline models
├── train_optic_disc_unet.py     step 1: optic disc network
├── train_fovea.py               step 2: fovea decoder
├── train_unet.py                step 3: lesion segmentation network
├── train_paragraph.py           step 4: PARAGraph
├── grading_dataloader.py        grading data (fixed format), splits, augmentation
├── segmentation_dataloader.py   lesion, optic disc and fovea data (fixed format)
├── scripts/                     run scripts with the experiment configurations
├── evaluation/                  testing, robustness, faithfulness, segmentation metrics
├── baselines/                   training scripts of the baseline models
├── visualization/               figures
└── docs/                        notes on the robustness experiments and the re-implemented baselines
```

**Run every command from the repository root**, for example
`python evaluation/test_paragraph.py ...` or `bash scripts/train_paragraph.sh data/ddr`.
Default paths such as `./checkpoints/` are relative to it.

## Setup

The experiments were run with Python 3.11 and PyTorch 2.8.

```bash
pip install -r requirements.txt
```

Install the `torch` and `torchvision` builds that match your CUDA version first.

## Data

No images are included. The code does not know any dataset by name. It reads data from a
folder in one fixed format, so any dataset works once it is arranged this way. Pass the
folder with `--data_root`.

### Grading data

Used by PARAGraph, the baselines and the evaluation scripts.

```
<data_root>/
├── images/        image files, sub-folders allowed
├── train.csv
├── test.csv       optional, see below
└── val.csv        optional
```

Each CSV has a header and two columns:

```
image,label
train/007-0004-000.jpg,0
train/007-0025-100.jpg,2
```

| Column | Meaning |
|---|---|
| `image` | Path of the image relative to `<data_root>/images/`. An absolute path also works |
| `label` | DR grade, an integer from 0 to 4: no DR, mild, moderate, severe, proliferative |

Images of any size are accepted. They are resized to a square of `--image_size` pixels.

How the splits are formed:

| Folder contains | Without `--split_seed` | With `--split_seed S` |
|---|---|---|
| `train.csv` and `test.csv` | The files are used as they are | The test set stays fixed. `train.csv` and `val.csv` are pooled and re-split 80:20 into train and validation |
| only `train.csv` | 20% is held out as the test set, seed 42 | 20% is held out as the test set, seed `S` |

All splits are stratified by label. They depend only on the order of the CSV rows and the
seed. `--val_ratio` and `--test_ratio` change the two shares.

### Segmentation data

Used by the three segmentation training scripts. All three tasks read their images from
the same place and differ only in the annotation.

```
<data_root>/
├── images/
│   ├── train/     image files
│   ├── test/
│   └── val/       optional. Without it, the test split is used for validation
├── masks/
│   └── <split>/<CLASS>/<image name without extension>.png
└── fovea_<split>.csv
```

| Task | Script | Annotation |
|---|---|---|
| Optic disc | `train_optic_disc_unet.py` | `masks/<split>/OD/`. Only images that have an OD mask are used |
| Fovea | `train_fovea.py` | `fovea_<split>.csv` with the columns `image,x,y`: file name and the fovea position in pixels of the original image |
| Lesions | `train_unet.py` | `masks/<split>/MA/`, `HE/`, `EX/` and `SE/`. Every image is used. A missing mask file means the image has no lesion of that class |

Masks are binary images of the same size as their image. Every pixel above 0 belongs to
the class. PNG, TIFF and JPEG masks are accepted.

One folder can serve several tasks, or each task can have its own folder. In the paper the
optic disc and fovea networks are trained on IDRiD, and the lesion network on the lesion
masks of IDRiD, DDR and FGADR.

## Checkpoints

This repository contains code only. No trained weights are distributed. All networks are
trained with the pipeline in the next section.

PARAGraph training needs the two frozen segmentation checkpoints at these paths. Steps 1
to 3 of the pipeline write them there:

```
checkpoints/optic_unet_resnet50/best_model.pth
checkpoints/lesion_unet_swint/best_model.pth
```

A trained PARAGraph checkpoint is self-contained. It embeds the segmentation and encoder
weights, so evaluation needs only that one file.

## Training

The full pipeline has four steps. Steps 1 to 3 train the segmentation networks once.
Step 4 trains PARAGraph on top of them. The optic disc and fovea network has a ResNet-50
encoder. The lesion network has a Swin-T encoder.

### 1. Optic disc network

```bash
python train_optic_disc_unet.py --data_root data/optic_disc --encoder resnet50 \
    --image_size 1024 --epochs 100 --save_dir ./checkpoints/optic_unet_od
```

This trains the encoder and the OD decoder of `OpticUNet` on the OD masks.

### 2. Fovea decoder

```bash
python train_fovea.py --data_root data/fovea --encoder resnet50 --image_size 1024 --epochs 50 \
    --od_checkpoint ./checkpoints/optic_unet_od/best_model.pth \
    --save_dir ./checkpoints/optic_unet_resnet50
```

This loads the network of step 1, freezes its encoder and OD decoder, and trains only the
fovea decoder. The target is a Gaussian heat-map around the annotated fovea centre.

The checkpoint written here holds the encoder and both decoders. It is the file that
PARAGraph loads through `--optic_checkpoint`.

### 3. Lesion segmentation network

```bash
python train_unet.py --data_root data/lesions --encoder swin --image_size 1024 \
    --epochs 200 --batch_size 6 --accum_steps 4 --lr 2e-4 --weight_decay 1e-4 \
    --scheduler onecycle --use_amp --save_dir ./checkpoints/lesion_unet_swint
```

This trains the four-class lesion network (MA, HE, EX, SE).

### 4. PARAGraph

```bash
python train_paragraph.py --data_root data/ddr --split_seed 42 \
    --optic_checkpoint ./checkpoints/optic_unet_resnet50/best_model.pth \
    --lesion_checkpoint ./checkpoints/lesion_unet_swint/best_model.pth \
    --save_dir ./checkpoints/paragraph_ddr
```

The run writes `best_model.pth` and `test_results.json` into the save directory.
`python train_paragraph.py --help` lists every option.

`scripts/train_paragraph.sh` holds the full configuration of the paper and repeats the
training over five split seeds:

```bash
bash scripts/train_paragraph.sh data/ddr
bash scripts/train_paragraph.sh data/aptos
SEEDS="42 0 1" EPOCHS=30 bash scripts/train_paragraph.sh data/ddr ddr_short
```

The first argument is the data folder. The optional second one names the outputs.

### Ablations

```bash
bash scripts/run_ablation.sh data/ddr            # all configurations, three runs each
bash scripts/run_ablation.sh data/aptos L0 L1    # only the named configurations
```

Each row of the ablation table is one flag of `train_paragraph.py`:

| Flag | What is removed |
|---|---|
| `--no_lesion_unet` | All lesion nodes. The graph keeps only the OD and global nodes |
| `--lesion_direct` | Category and region nodes. Lesions connect straight to the hubs |
| `--no_region` | Region nodes |
| `--no_global_node` | The global node at the fovea |
| `--no_od_posbias` | The OD positional bias in the Swin encoder |
| `--no_global_logits` | The global output branch |
| `--no_prior_fusion` | The fusion of position, type and area priors into node features |
| `--no_gating` | Confidence gating |

## Evaluation

```bash
python evaluation/test_paragraph.py --checkpoints path/to/best_model.pth \
    --data_root data/ddr --split test
```

The script reports accuracy, quadratic weighted kappa, per-class accuracy and the confusion
matrix. Pass several checkpoints to compare them. To test on another dataset, point
`--data_root` at its folder. For a folder without `test.csv`, pass the `--split_seed` of
the training run so that the same test part is held out.

`evaluation/eval_lesion_segmentation.py` reports pixel-level precision, recall, F1 and IoU
of a lesion segmentation checkpoint on a lesion data folder.

## Robustness and interpretability

[docs/credibility.md](docs/credibility.md) gives the commands for two inference-only
analyses on a trained checkpoint:

- **Segmentation degradation.** Missed, misclassified and false-positive lesions are
  injected into the segmentation output at increasing rates.
- **Lesion faithfulness.** Lesions are deleted or inserted in order of attributed
  importance and compared with a random order.

In `visualization/`, `visualize_ig.py` draws the node-level attributions of one image on
top of it, and `visualize_ig_dataset.py` summarises them over a test set.

## Baselines

The training scripts are in `baselines/`. They take the same `--data_root` folder as
PARAGraph.

```bash
bash scripts/train_baselines.sh swinL data/ddr     # also: vitL, green
bash scripts/train_mdgnet.sh data/ddr
LESION_ROOT=data/lesions bash scripts/train_pranet.sh data/ddr
```

- **Image classifiers:** `train_swin.py`, `train_vit.py`, `train_densenet.py`,
  `train_efficientnet.py` and `train_cabnet.py`. `scripts/train_baselines.sh` holds the
  Swin-L and ViT-L configurations.
- **Swin with the OD positional bias only:** `train_swin_od.py`.
- **GREEN and the other DGDR methods:** `train_dgdr.py`. It imports the
  [DGDR](https://github.com/chehx/DGDR) code, which must be cloned next to this repository
  as `../DGDR`.
- **PRANet and MDGNet:** neither has public code. Both are re-implemented from the papers.
  [docs/baselines_pranet_mdgnet.md](docs/baselines_pranet_mdgnet.md) lists every detail
  that had to be inferred.

## Acknowledgements

- `Models/hacdr_lesion_unet.py` is a PyTorch port of the HACDR-Net backbone (AAAI 2024).
- `Models/cabnet_classifier.py` follows CABNet (He et al.).
- `baselines/train_dgdr.py` wraps [DGDR / GDRNet](https://github.com/chehx/DGDR) (MICCAI 2023).

## Citation

If you use this code, please cite the paper:

> PARAGraph: Pathology-Anatomy-Aware Hierarchical Graph for Diabetic Retinopathy Grading.
> IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2026.

A BibTeX entry will be added when the proceedings are online.

## License

Released under the MIT License. See [LICENSE](LICENSE).
