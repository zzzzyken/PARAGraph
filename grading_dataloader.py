"""
Grading data in one fixed format.

A dataset is a folder:

    <data_root>/
    ├── images/        image files, sub-folders allowed
    ├── train.csv      header "image,label"
    ├── test.csv       header "image,label"   (optional, see below)
    └── val.csv        header "image,label"   (optional)

`image` is a path relative to <data_root>/images/ (an absolute path also works).
`label` is the DR grade, an integer from 0 to 4.

How the three splits are formed:

  * test.csv present: it is the test set, unchanged.
        split_seed is None  -> train = train.csv, val = val.csv (if present)
        split_seed is given -> train.csv and val.csv are pooled and re-split into
                               train / val (stratified, val_ratio, that seed)
  * test.csv absent:  a stratified test_ratio part of train.csv is held out as the
                      test set, with split_seed (42 if None).

The splits are stratified by label and depend only on the order of the rows and the seed.
"""

import csv
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============ labels ============
def map_dr_grade_to_stage(grade: int) -> Optional[int]:
    """Five DR grades -> three stages: 0 healthy, 1 NPDR (grades 1-3), 2 PDR (grade 4)."""
    if grade == 0:
        return 0
    elif grade in [1, 2, 3]:
        return 1
    elif grade == 4:
        return 2
    raise ValueError(f"Unknown DR grade: {grade}")


STAGE_NAMES = ['Healthy', 'NPDR', 'PDR']
GRADE_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
SPLIT_FILES = {'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'}


# ============ reading the fixed format ============
def read_split_csv(data_root: str, split: str) -> List[Tuple[str, int]]:
    """Rows of <data_root>/<split>.csv as (absolute image path, grade 0-4), in file order."""
    path = os.path.join(data_root, SPLIT_FILES[split])
    image_dir = os.path.join(data_root, 'images')
    samples = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {'image', 'label'} <= {c.strip() for c in reader.fieldnames}:
            raise ValueError(f'{path}: the header must contain the columns "image" and "label"')
        for row_no, row in enumerate(reader, start=2):
            row = {k.strip(): (v or '').strip() for k, v in row.items() if k}
            if not row.get('image'):
                continue
            try:
                grade = int(row['label'])
            except ValueError:
                raise ValueError(f'{path}, line {row_no}: label "{row["label"]}" is not an integer')
            if not 0 <= grade <= 4:
                raise ValueError(f'{path}, line {row_no}: label {grade} is outside 0-4')
            image = row['image']
            samples.append((image if os.path.isabs(image) else os.path.join(image_dir, image), grade))
    if not samples:
        raise ValueError(f'{path} has no rows')
    return samples


def has_split(data_root: str, split: str) -> bool:
    return os.path.isfile(os.path.join(data_root, SPLIT_FILES[split]))


def stratified_split(samples: List[Tuple[str, int]], ratio: float, seed: int) -> Tuple[List, List]:
    """Split `samples` into (held_out, rest), stratified by label.

    Labels are visited in ascending order. Within a label the rows are shuffled with one
    random.Random(seed) and the first max(1, int(n * ratio)) rows are held out.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)
    held_out, rest = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = max(1, int(len(group) * ratio))
        held_out.extend(group[:n])
        rest.extend(group[n:])
    return held_out, rest


def load_splits(
    data_root: str,
    split_seed: Optional[int] = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    verbose: bool = True,
) -> Dict[str, List[Tuple[str, int]]]:
    """Sample lists {'train', 'test', optionally 'val'} for a dataset folder (rules in the module docstring)."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'data_root does not exist: {data_root}')
    if not has_split(data_root, 'train') and not has_split(data_root, 'test'):
        raise FileNotFoundError(f'{data_root} has neither train.csv nor test.csv')

    splits: Dict[str, List[Tuple[str, int]]] = {}
    if has_split(data_root, 'test'):
        splits['test'] = read_split_csv(data_root, 'test')
        if has_split(data_root, 'train'):
            train = read_split_csv(data_root, 'train')
            val = read_split_csv(data_root, 'val') if has_split(data_root, 'val') else []
            if split_seed is not None:
                splits['val'], splits['train'] = stratified_split(train + val, val_ratio, split_seed)
            else:
                splits['train'] = train
                if val:
                    splits['val'] = val
    else:
        seed = 42 if split_seed is None else split_seed
        splits['test'], splits['train'] = stratified_split(read_split_csv(data_root, 'train'), test_ratio, seed)
        if has_split(data_root, 'val'):
            splits['val'] = read_split_csv(data_root, 'val')

    if verbose:
        name = os.path.basename(os.path.normpath(data_root))
        print(f"[{name}] " + ', '.join(f"{k}: {len(v)} images" for k, v in splits.items())
              + (f"  (split_seed={split_seed})" if split_seed is not None else ''))
    return splits


# ============ dataset ============
class GradingDataset(Dataset):
    """A list of (image path, grade) pairs. Returns the same sample dictionary for every dataset."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        image_size: int = 224,
        use_3_stages: bool = True,
        name: str = 'dataset',
    ):
        super().__init__()
        self.transform = transform
        self.image_size = image_size
        self.use_3_stages = use_3_stages
        self.name = name
        self.samples = [(p, map_dr_grade_to_stage(g) if use_3_stages else g) for p, g in samples]
        if len(self.samples) == 0:
            raise ValueError(f'No samples for {name}')

    def __len__(self):
        return len(self.samples)

    def get_labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, label = self.samples[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        transform = self.transform or get_val_transform(self.image_size)
        image = transform(image=image)['image']
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': os.path.basename(image_path),
            'image_path': image_path,
            'dataset': self.name,
        }


def build_dataset(
    data_root: str,
    split: str,
    transform: Optional[Callable] = None,
    image_size: int = 224,
    use_3_stages: bool = True,
    split_seed: Optional[int] = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    verbose: bool = False,
) -> GradingDataset:
    """One split ('train', 'val' or 'test') of a dataset folder as a GradingDataset."""
    splits = load_splits(data_root, split_seed, val_ratio, test_ratio, verbose=verbose)
    if split not in splits:
        raise ValueError(f"{data_root} has no '{split}' split (available: {sorted(splits)})")
    return GradingDataset(splits[split], transform, image_size, use_3_stages,
                          name=os.path.basename(os.path.normpath(data_root)))


class BalancedBatchSampler(Sampler):
    """每个batch保证每个类至少有一个样本，剩余槽随机填充。"""

    def __init__(self, labels: List[int], batch_size: int, num_classes: int):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_indices = {c: [] for c in range(num_classes)}
        for idx, lbl in enumerate(labels):
            if lbl in self.class_indices:
                self.class_indices[lbl].append(idx)
        self.n_batches = len(labels) // batch_size

    def __iter__(self):
        import random as _random
        pools = {c: _random.sample(idxs, len(idxs)) for c, idxs in self.class_indices.items() if idxs}
        ptrs  = {c: 0 for c in pools}

        def _next(c):
            if ptrs[c] >= len(pools[c]):
                pools[c] = _random.sample(self.class_indices[c], len(self.class_indices[c]))
                ptrs[c] = 0
            idx = pools[c][ptrs[c]]
            ptrs[c] += 1
            return idx

        active_classes = [c for c in range(self.num_classes) if self.class_indices.get(c)]
        for _ in range(self.n_batches):
            batch = [_next(c) for c in active_classes]          # 1 per class
            extra = self.batch_size - len(active_classes)
            for _ in range(extra):
                batch.append(_next(_random.choice(active_classes)))
            _random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches * self.batch_size


# ============ 数据增强 ============
def get_train_transform(image_size: int = 224):
    """训练时的数据增强"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-45, 45),
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=4.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform(image_size: int = 224):
    """验证/测试时的变换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_strong_augmentation(image_size: int = 224):
    """分割安全的增强：不破坏 OD/病变区域，避免 CoarseDropout、Shear、强噪声。"""
    return A.Compose([
        A.Resize(image_size, image_size),
        # 几何变换 (冻结分割模型可处理翻转/小旋转/小缩放)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-30, 30),
            border_mode=0,
            p=0.5
        ),
        # 颜色增强 (中等强度，不破坏 OD/病变可见度)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ], p=0.7),
        # 轻度模糊 (不能太强，会掩盖微小病变)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def compute_sample_weights(dataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weight of every sample, for WeightedRandomSampler."""
    labels = dataset.get_labels()
    class_counts = torch.zeros(num_classes)
    for label in labels:
        class_counts[label] += 1

    class_weights = 1.0 / class_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * num_classes

    sample_weights = torch.tensor([class_weights[label] for label in labels])

    print(f"[Oversample] Class counts: {class_counts.tolist()}")
    print(f"[Oversample] Class weights: {class_weights.tolist()}")

    return sample_weights


# ============ DataLoader factory ============
def create_grading_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_3_stages: bool = True,
    use_oversample: bool = False,
    use_strong_aug: bool = False,
    use_balanced_sampler: bool = False,
    split_seed: Optional[int] = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Dict[str, DataLoader]:
    """
    DataLoaders {'train', 'test', optionally 'val'} for a dataset folder in the fixed format.

    Args:
        data_root: dataset folder (see the module docstring)
        use_3_stages: 3 stages (True) or the 5 grades (False)
        use_oversample: WeightedRandomSampler with inverse class frequency
        use_balanced_sampler: every batch holds at least one sample of every class
        use_strong_aug: stronger training augmentation
        split_seed, val_ratio, test_ratio: see the module docstring
    """
    dataloaders = {}
    num_classes = 3 if use_3_stages else 5
    name = os.path.basename(os.path.normpath(data_root))
    splits = load_splits(data_root, split_seed, val_ratio, test_ratio)

    if 'train' in splits:
        train_transform = get_strong_augmentation(image_size) if use_strong_aug else get_train_transform(image_size)
        train_dataset = GradingDataset(splits['train'], train_transform, image_size, use_3_stages, name)

        if use_balanced_sampler:
            print("\n[BalancedSampler] Building per-class index pools...")
            sampler = BalancedBatchSampler(train_dataset.get_labels(), batch_size, num_classes)
            print(f"[BalancedSampler] {sampler.n_batches} batches/epoch, "
                  f"{num_classes} classes guaranteed per batch")
            dataloaders['train'] = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        elif use_oversample:
            print("\n[Oversample] Computing sample weights for balanced sampling...")
            sample_weights = compute_sample_weights(train_dataset, num_classes)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            dataloaders['train'] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True
            )
        else:
            dataloaders['train'] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True
            )

    for split in ('val', 'test'):
        if split in splits:
            dataloaders[split] = DataLoader(
                GradingDataset(splits[split], get_val_transform(image_size), image_size, use_3_stages, name),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

    return dataloaders


# ============ 统计工具 ============
def print_dataset_statistics(dataloaders: Dict[str, DataLoader], use_3_stages: bool = True):
    """打印数据集统计信息"""
    stage_names = STAGE_NAMES if use_3_stages else GRADE_NAMES

    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)

    for split_name, dataloader in dataloaders.items():
        print(f"\n{split_name.upper()} Set:")
        print(f"  Total samples: {len(dataloader.dataset)}")

        # 统计各类别数量
        label_counts = {}
        for label in dataloader.dataset.get_labels():
            label_counts[label] = label_counts.get(label, 0) + 1

        print("  Class distribution:")
        for label, count in sorted(label_counts.items()):
            name = stage_names[label] if label < len(stage_names) else f"Class {label}"
            pct = count / len(dataloader.dataset) * 100
            print(f"    {name}: {count} ({pct:.1f}%)")
