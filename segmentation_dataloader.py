"""
Segmentation and localisation data in one fixed format.

A dataset is a folder. Every task reads its images from the same place:

    <data_root>/
    ├── images/
    │   ├── train/      image files
    │   ├── test/
    │   └── val/        (optional; the test split is used for validation when it is missing)
    ├── masks/          lesion and optic disc tasks
    │   └── <split>/<CLASS>/<image name without extension>.png   (.tif, .jpg ... also work)
    └── fovea_<split>.csv   fovea task, header "image,x,y" (pixel position in the original image)

Masks are binary: every pixel above 0 belongs to the class.

  * Lesions: CLASS is MA, HE, EX or SE. Every image of the split is used.
    A missing mask file means the image has no lesion of that class.
  * Optic disc: CLASS is OD. Only images that have an OD mask are used.
  * Fovea: only images listed in fovea_<split>.csv are used.
"""

import csv
import os
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


LESION_NAMES = ['MA', 'HE', 'EX', 'SE']
BASIC_LESION_NAMES = LESION_NAMES          # earlier name, still imported by some scripts
LESION_CLASSES = dict(enumerate(LESION_NAMES))
LESION_COLORS = {
    'MA': (255, 0, 0),
    'HE': (0, 255, 0),
    'EX': (255, 255, 0),
    'SE': (0, 0, 255),
}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


# ============ reading the fixed format ============
def list_images(data_root: str, split: str) -> List[str]:
    """Image files of <data_root>/images/<split>/, sorted by name."""
    folder = os.path.join(data_root, 'images', split)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'{folder} does not exist')
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        raise ValueError(f'No images in {folder}')
    return [os.path.join(folder, f) for f in files]


def has_split(data_root: str, split: str) -> bool:
    return os.path.isdir(os.path.join(data_root, 'images', split))


def index_masks(data_root: str, split: str, class_name: str) -> Dict[str, str]:
    """{image name without extension: mask path} for one class of one split."""
    folder = os.path.join(data_root, 'masks', split, class_name)
    if not os.path.isdir(folder):
        return {}
    return {os.path.splitext(f)[0]: os.path.join(folder, f)
            for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTENSIONS)}


def load_binary_mask(path: str, height: int, width: int) -> np.ndarray:
    """Mask file -> float32 [height, width] array of 0 and 1."""
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.float32)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


# ============ masks: lesions and optic disc ============
class MaskDataset(Dataset):
    """
    Images with one binary mask channel per class.

    require_mask=False (lesions): every image is used, missing masks are empty.
    require_mask=True (optic disc): only images with a mask of every class are used.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        classes: Sequence[str] = tuple(LESION_NAMES),
        transform: Optional[Callable] = None,
        image_size: int = 512,
        require_mask: bool = False,
        return_path: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.classes = list(classes)
        self.transform = transform
        self.image_size = image_size
        self.return_path = return_path
        self.name = os.path.basename(os.path.normpath(data_root))

        self.mask_index = {c: index_masks(data_root, split, c) for c in self.classes}
        self.image_paths = list_images(data_root, split)
        if require_mask:
            self.image_paths = [p for p in self.image_paths
                                if all(os.path.splitext(os.path.basename(p))[0] in self.mask_index[c]
                                       for c in self.classes)]
            if not self.image_paths:
                raise ValueError(f'No image of {data_root}/images/{split} has masks for {self.classes}')

        print(f"[{self.name}] {len(self.image_paths)} images in the {split} split, classes {self.classes}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        H, W = image.shape[:2]
        mask = np.zeros((H, W, len(self.classes)), dtype=np.float32)
        for i, c in enumerate(self.classes):
            path = self.mask_index[c].get(image_id)
            if path is not None:
                mask[:, :, i] = load_binary_mask(path, H, W)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if mask.ndim == 3 and mask.shape[-1] == len(self.classes):   # albumentations returns HWC masks
                mask = mask.permute(2, 0, 1).contiguous()
            mask = mask.float()
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            if mask.ndim == 2:
                mask = mask[:, :, None]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        output = {
            'image': image,                # [3, H, W]
            'mask': mask,                  # [number of classes, H, W]
            'image_id': image_id,
            'dataset': self.name,
        }
        if self.return_path:
            output['image_path'] = image_path
        return output


# ---- lesions ----
def get_lesion_train_transform(image_size: int = 512):
    """训练时的数据增强"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2
            ),
            A.CLAHE(clip_limit=4.0),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_lesion_val_transform(image_size: int = 512):
    """验证/测试时的变换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_lesion_dataloaders(
    data_root: str,
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """DataLoaders {'train', 'val', 'test'} for the four-class lesion task (masks MA, HE, EX, SE)."""
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        MaskDataset(data_root, 'train', LESION_NAMES, get_lesion_train_transform(image_size), image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_split = 'val' if has_split(data_root, 'val') else 'test'
    for name, split in (('val', val_split), ('test', 'test')):
        if has_split(data_root, split):
            dataloaders[name] = DataLoader(
                MaskDataset(data_root, split, LESION_NAMES, get_lesion_val_transform(image_size), image_size),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
    return dataloaders


# ---- optic disc ----
def get_optic_disc_train_transform(image_size: int = 1022):
    """训练时的数据增强"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            p=0.5,
            border_mode=0
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=4.0),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_optic_disc_val_transform(image_size: int = 1022):
    """验证/测试时的变换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_optic_disc_dataloaders(
    data_root: str,
    image_size: int = 1022,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """DataLoaders {'train', 'val', 'test'} for the optic disc task (masks OD)."""
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        MaskDataset(data_root, 'train', ['OD'], get_optic_disc_train_transform(image_size), image_size,
                    require_mask=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    dataloaders['test'] = DataLoader(
        MaskDataset(data_root, 'test', ['OD'], get_optic_disc_val_transform(image_size), image_size,
                    require_mask=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataloaders['val'] = dataloaders['test']
    if has_split(data_root, 'val'):
        dataloaders['val'] = DataLoader(
            MaskDataset(data_root, 'val', ['OD'], get_optic_disc_val_transform(image_size), image_size,
                        require_mask=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return dataloaders


# ============ fovea ============
def read_fovea_csv(data_root: str, split: str) -> Dict[str, tuple]:
    """{image name without extension: (x, y)} from <data_root>/fovea_<split>.csv."""
    path = os.path.join(data_root, f'fovea_{split}.csv')
    coordinates = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {'image', 'x', 'y'} <= {c.strip() for c in reader.fieldnames}:
            raise ValueError(f'{path}: the header must contain the columns "image", "x" and "y"')
        for row in reader:
            row = {k.strip(): (v or '').strip() for k, v in row.items() if k}
            if not row.get('image'):
                continue
            try:
                coordinates[os.path.splitext(row['image'])[0]] = (int(float(row['x'])), int(float(row['y'])))
            except ValueError:
                continue
    return coordinates


class FoveaDataset(Dataset):
    """Images with the fovea centre, returned as a Gaussian heat-map and as coordinates."""

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 1024,
        heatmap_sigma: float = 10.0,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.name = os.path.basename(os.path.normpath(data_root))

        self.coordinates = read_fovea_csv(data_root, split)
        self.samples = []
        for image_path in list_images(data_root, split):
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            if image_id in self.coordinates:
                self.samples.append({
                    'image_path': image_path,
                    'image_id': image_id,
                    'fovea_x': self.coordinates[image_id][0],
                    'fovea_y': self.coordinates[image_id][1],
                })
        if not self.samples:
            raise ValueError(f'fovea_{split}.csv of {data_root} matches no image in images/{split}')

        print(f"[{self.name}] {len(self.samples)} images with a fovea position in the {split} split")

    def __len__(self):
        return len(self.samples)

    def _generate_heatmap(
        self,
        center_x: float,
        center_y: float,
        height: int,
        width: int,
        sigma: float
    ) -> np.ndarray:
        """
        生成以 (center_x, center_y) 为中心的高斯热图

        Args:
            center_x, center_y: 中心点坐标 (相对于 width, height)
            height, width: 热图尺寸
            sigma: 高斯标准差

        Returns:
            heatmap: [H, W] 归一化的高斯热图
        """
        x = np.arange(0, width, 1, dtype=np.float32)
        y = np.arange(0, height, 1, dtype=np.float32)
        x, y = np.meshgrid(x, y)

        heatmap = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

        return heatmap

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 加载图像
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        orig_h, orig_w = image.shape[:2]

        # 原始 Fovea 坐标
        fovea_x = sample['fovea_x']
        fovea_y = sample['fovea_y']

        # 创建 keypoint 用于 albumentations 变换
        keypoints = [(fovea_x, fovea_y)]

        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']

            # 获取变换后的坐标
            if len(keypoints) > 0:
                new_x, new_y = keypoints[0]
            else:
                # 如果 keypoint 被裁剪掉了，使用图像中心
                new_x, new_y = self.image_size / 2, self.image_size / 2
        else:
            # 如果没有 transform，手动 resize
            new_x = fovea_x * self.image_size / orig_w
            new_y = fovea_y * self.image_size / orig_h

        # 生成热图
        heatmap = self._generate_heatmap(
            center_x=new_x,
            center_y=new_y,
            height=self.image_size,
            width=self.image_size,
            sigma=self.heatmap_sigma
        )

        # 转为 tensor
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]

        # 归一化坐标 (0-1)
        norm_x = new_x / self.image_size
        norm_y = new_y / self.image_size

        return {
            'image': image,
            'heatmap': heatmap,
            'fovea_x': torch.tensor(new_x, dtype=torch.float32),
            'fovea_y': torch.tensor(new_y, dtype=torch.float32),
            'fovea_x_norm': torch.tensor(norm_x, dtype=torch.float32),
            'fovea_y_norm': torch.tensor(norm_y, dtype=torch.float32),
            'orig_fovea_x': sample['fovea_x'],
            'orig_fovea_y': sample['fovea_y'],
            'orig_size': (orig_h, orig_w),
            'image_id': sample['image_id']
        }


def get_fovea_train_transform(image_size: int = 1024):
    """训练时的数据增强"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            p=0.5,
            border_mode=0
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=4.0),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_fovea_val_transform(image_size: int = 1024):
    """验证/测试时的变换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def create_fovea_dataloaders(
    data_root: str,
    image_size: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    heatmap_sigma: float = 10.0,
) -> Dict[str, DataLoader]:
    """DataLoaders {'train', 'val', 'test'} for the fovea task."""
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        FoveaDataset(data_root, 'train', get_fovea_train_transform(image_size), image_size, heatmap_sigma),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    dataloaders['test'] = DataLoader(
        FoveaDataset(data_root, 'test', get_fovea_val_transform(image_size), image_size, heatmap_sigma),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataloaders['val'] = dataloaders['test']
    if has_split(data_root, 'val') and os.path.isfile(os.path.join(data_root, 'fovea_val.csv')):
        dataloaders['val'] = DataLoader(
            FoveaDataset(data_root, 'val', get_fovea_val_transform(image_size), image_size, heatmap_sigma),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return dataloaders


# ============ visualisation helper ============
def visualize_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    pred_mask: torch.Tensor = None,
    image_id: str = None,
    save_path: str = None
):
    """
    可视化样本及其mask
    
    Args:
        image: [3, H, W] 归一化后的图像
        mask: [4, H, W] ground truth mask
        pred_mask: [4, H, W] 预测的mask (可选)
        image_id: 图像ID
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image.cpu() * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    
    mask = mask.cpu().numpy()
    
    n_cols = 6 if pred_mask is not None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 各类lesion的GT mask
    lesion_names = ['MA', 'HE', 'EX', 'SE']
    colors = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]
    
    for i, (name, color) in enumerate(zip(lesion_names, colors)):
        overlay = image.copy()
        m = mask[i]
        
        for c in range(3):
            overlay[:, :, c] = np.where(m > 0.5, 
                                        overlay[:, :, c] * 0.5 + color[c] * 0.5,
                                        overlay[:, :, c])
        
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f'GT: {name}')
        axes[i + 1].axis('off')
    
    # 预测mask (如果有)
    if pred_mask is not None:
        pred_mask = pred_mask.cpu().numpy()
        combined_overlay = image.copy()
        
        for i, color in enumerate(colors):
            m = pred_mask[i]
            for c in range(3):
                combined_overlay[:, :, c] = np.where(
                    m > 0.5,
                    combined_overlay[:, :, c] * 0.5 + color[c] * 0.5,
                    combined_overlay[:, :, c]
                )
        
        axes[-1].imshow(combined_overlay)
        axes[-1].set_title('Prediction (All)')
        axes[-1].axis('off')
    
    if image_id:
        fig.suptitle(f'Image: {image_id}', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============ 测试 ============
