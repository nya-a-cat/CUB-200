# custom_transforms.py
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
import random

def get_augmentation_transforms(size=224):
    common_transforms = [
        T.ToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((256,256)),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    aug1 = T.Compose([
        *common_transforms,
        # T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        T.RandomHorizontalFlip(p=1.0)
    ])
    aug2 = T.Compose([
        *common_transforms,
        # T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        T.RandomRotation(degrees=(30, 30))
    ])
    return aug1, aug2

def get_inverse_transforms():
    inv_aug1 = T.Compose([
        T.RandomHorizontalFlip(p=1.0)  # 近似逆向
    ])
    inv_aug2 = T.Compose([
        T.RandomRotation(degrees=(-30, -30)),  # 近似逆向
        T.RandomHorizontalFlip(p=1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-10, 10, -10, 10))
    ])
    return inv_aug1, inv_aug2


# 如果你想真正地“记录随机参数并逆向”的话，可以再写一个自定义类：
class Random:
    """
    在 forward 时随机采样参数并应用到图像/特征；
    在 inverse 时使用已知参数的反向值再应用到图像/特征。
    可在 DataLoader 或训练循环中使用。
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, fill=0):
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill = fill

    def generate_random_params(self):
        angle = random.uniform(self.degrees[0], self.degrees[1])

        if self.translate is not None:
            max_dx, max_dy = self.translate
            translate_x = random.uniform(-max_dx, max_dx)
            translate_y = random.uniform(-max_dy, max_dy)
            translations = (translate_x, translate_y)
        else:
            translations = (0, 0)

        if self.scale is not None:
            min_s, max_s = self.scale
            scale = random.uniform(min_s, max_s)
        else:
            scale = 1.0

        if self.shear is not None:
            max_shear_x, max_shear_y = self.shear
            shear_x = random.uniform(-max_shear_x, max_shear_x)
            shear_y = random.uniform(-max_shear_y, max_shear_y)
            shear_params = [shear_x, shear_y]
        else:
            shear_params = [0.0, 0.0]

        return angle, translations, scale, shear_params

    def forward(self, tensor):
        """
        对单个 (C,H,W) 的张量执行随机仿射变换。
        返回 (变换后的tensor, 当前使用的params)。
        """
        angle, translations, scale, shear_params = self.generate_random_params()
        # 在 v2 的 F.affine 里，可以对 Tensor 直接操作(>=0.13)，无需转PIL。
        transformed = F.affine(
            tensor, angle=angle,
            translate=translations,
            scale=scale,
            shear=shear_params
        )
        params = {
            'angle': angle,
            'translations': translations,
            'scale': scale,
            'shear': shear_params
        }
        return transformed, params

    def inverse(self, tensor, params):
        """
        给定前向使用的 params，对张量执行“逆向”仿射变换。
        """
        angle = -params['angle']
        tx, ty = params['translations']
        translations = (-tx, -ty)
        scale = 1.0 / params['scale']
        shear = [-s for s in params['shear']]

        inverted = F.affine(
            tensor, angle=angle,
            translate=translations,
            scale=scale,
            shear=shear
        )
        return inverted
