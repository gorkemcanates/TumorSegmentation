__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from torch.optim import Adam
from trainer.trainer import MainTrainer
from Dataset.Datasets import LITSDataset
from model.res_unet_plus import ResUnetPlus
from losses.loss import *
from metrics.segmentation import IoU, DiceScore
from transforms.transforms import Transforms
from writer.writer import TensorboardWriter
import warnings

warnings.filterwarnings("ignore")

class Parameters:
    def __init__(self):
        self.experiment = 'LITS/'
        self.file = 'ResUnetplus_class_4_lr_e-4/'
        self.load_file = 'ResUnetplus_lr001/'
        self.train_data_dir = 'Dataset\images/'
        self.train_mask_dir = 'Dataset\masks/'
        self.LOGDIR = f'runs/' + self.experiment + self.file
        self.FIG_PATH = 'RESULTS/' + self.experiment + self.file + 'images/'
        self.result_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'metrics/'
        self.model_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'models/'
        self.model_LOADPATH = 'RESULTS/' + self.experiment + self.load_file + 'models/'
        self.METRIC_CONDITION = IoU.__name__.lower()
        self.TO_TENSORBOARD = True
        self.VALIDATION = True
        self.PRETRAINED = False
        self.DEBUG = False
        self.SHUFFLE = True
        self.TRANSFORM = True
        self.DEVICE = 'cuda'

 
class HyperParameters:
    def __init__(self):
        self.NUM_EPOCHS = 200
        self.LEARNING_RATE = 0.00001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.IN_CHANNELS = 3
        self.NUM_CLASSES = 3
        self.FILTER_COEFF = 0.5
        self.train_batch_size = 4
        self.test_batch_size = 4
        self.resize_shape = 512
        # self.BACKBONE = 'ResNet-34'
        # self.ATTENTION = 'SE'
        self.NORM = 'gn'
        self.METRIC_CONDITION = 'max'



class MAIN:
    def __init__(self):
        self.params = Parameters()
        self.hyperparams = HyperParameters()

        self.model = ResUnetPlus(in_features=self.hyperparams.IN_CHANNELS,
                                out_features=self.hyperparams.NUM_CLASSES,
                                k=self.hyperparams.FILTER_COEFF,
                                norm_type=self.hyperparams.NORM)

        self.metrics = [IoU(num_classes=self.hyperparams.NUM_CLASSES),
                        DiceScore(num_classes=self.hyperparams.NUM_CLASSES)
                        ]


        self.criterion = IoULoss(num_classes=self.hyperparams.NUM_CLASSES,
                                 reduction='mean')


        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.hyperparams.LEARNING_RATE,
                              betas=(self.hyperparams.beta1,
                                     self.hyperparams.beta2))

        self.transforms = Transforms(transform=self.params.TRANSFORM)


        self.dataset = LITSDataset(im_path=self.params.train_data_dir,
                                   mask_path=self.params.train_mask_dir,
                                   train_transform=self.transforms.train_transform,
                                   val_transform=self.transforms.val_transform,
                                   num_classes=self.hyperparams.NUM_CLASSES,
                                   debug=self.params.DEBUG
                                    )


        self.writer = TensorboardWriter(PATH=self.params.LOGDIR,
                                        fig_path=self.params.FIG_PATH,
                                        num_data=48)


        self.trainer = MainTrainer(model=self.model,
                                   params=self.params,
                                   hyperparams=self.hyperparams,
                                   metrics=self.metrics,
                                   dataset=self.dataset,
                                   optimizer=self.optimizer,
                                   criterion=self.criterion,
                                   writer=self.writer
                                   if self.params.TO_TENSORBOARD else None
                                   )
        print(self.model)
        print(f'Total model parameters : '
              f'{sum(p.numel() for p in self.model.parameters())}')
        print(f'MODEL : {self.model._get_name()} ')
        print(f'CRITERION : {self.criterion._get_name()} ')
        print(f'FILTER COEFFICIENT: {self.hyperparams.FILTER_COEFF} ')
        print(f'BATCH SIZE : {self.hyperparams.train_batch_size} ')
        print(f'DEVICE : {self.params.DEVICE.upper()} ')
        print(torch.cuda_version)


    def run(self):
        self.trainer.fit()

    def validate(self):
        results = self.trainer.validate()
        return results


if __name__ == '__main__':
    trainer = MAIN()
    trainer.run()





