import imgaug.augmenters as iaa
import numpy as np
from detectron2.data.transforms import Augmentation, Transform

class LectureSpecificAugmentation(Augmentation):
    """Custom augmentations for lecture content"""
    
    def __init__(self):
        super().__init__()
        
        # Augmentations suitable for lecture slides
        self.augmenter = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
            iaa.Sometimes(0.3, iaa.contrast.LinearContrast((0.8, 1.2))),
            iaa.Sometimes(0.2, iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-5, 5)
            )),
            iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ])
    
    def get_transform(self, image):
        return LectureTransform(self.augmenter)

class LectureTransform(Transform):
    """Apply ImgAug transformations"""
    
    def __init__(self, augmenter):
        super().__init__()
        self.augmenter = augmenter
    
    def apply_image(self, img):
        return self.augmenter.augment_image(img)
    
    def apply_coords(self, coords):
        return coords  # ImgAug handles coordinate transformation internally