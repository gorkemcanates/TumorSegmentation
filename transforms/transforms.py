import albumentations as A
import albumentations.pytorch as Ap


class Transforms:
    def __init__(self, shape=None, transform=True):
        if shape is None:
            shape = [512, 512]
        if transform:
            self.train_transform = A.Compose([
                # A.Resize(shape[0], shape[1]),
                # A.CenterCrop(224, 224),
                A.ShiftScaleRotate(shift_limit=0.01,
                                   scale_limit=0.01,
                                   rotate_limit=180,
                                   p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.GridDistortion(p=0.5),
                    A.ElasticTransform(p=0.5),
                    A.OpticalDistortion(p=0.5,
                                        distort_limit=2,
                                        shift_limit=0.5)
                ],
                    p=0.2),
                A.GridDropout(p=0.2,
                              random_offset=True,
                              mask_fill_value=None),
                A.OneOf([A.GaussianBlur(p=0.5),
                         A.GlassBlur(p=0.5)],
                        p=0.2),
                A.ColorJitter(p=0.2,
                              brightness=0.4,
                              contrast=0.4,
                              saturation=0.4,
                              hue=0.4),
                A.GaussNoise(p=0.2),
                Ap.ToTensorV2()
            ])

            self.val_transform = A.Compose([
                Ap.ToTensorV2()
            ])
        else:
            self.train_transform = A.Compose([
                Ap.ToTensorV2()
            ])
            self.val_transform = A.Compose([
                Ap.ToTensorV2()
            ])




