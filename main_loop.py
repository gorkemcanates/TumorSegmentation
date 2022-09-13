__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from torch.optim import SGD
from trainer.trainer import MainTrainer
from Dataset.Datasets import ImageNetDataset, CIFARDataset
from model.backbones.resnet import ResNet
from losses.classification import CELoss
from metrics.classification import ErrorRate_T1, ErrorRate_T5
from transforms.transforms import Transforms, CIFARTransforms
from writer.writer import TensorboardWriter
import warnings

warnings.filterwarnings("ignore")


class Parameters:
    def __init__(self, fname, experiment):
        self.experiment = experiment
        # self.experiment = 'ImageNet/'
        self.file = fname
        self.load_file = 'ResNet-18-SE/'
        self.path = 'C:\GorkemCanAtes\PycharmProjects/SOTA/ImageNet/'
        self.train_data_dir = 'C:\GorkemCanAtes\PycharmProjects/SOTA/ImageNet/train_data/'
        self.train_label_dir = 'C:\GorkemCanAtes\PycharmProjects/SOTA/ImageNet/train_labels/'
        self.val_data_dir = 'C:\GorkemCanAtes\PycharmProjects/SOTA/ImageNet/validation_data/'
        self.val_label_dir = 'C:\GorkemCanAtes\PycharmProjects/SOTA/ImageNet/validation_labels/'
        self.LOGDIR = f'runs/' + self.experiment + self.file
        self.result_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'metrics/'
        self.model_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'models/'
        self.model_LOADPATH = 'RESULTS/' + self.experiment + self.load_file + 'models/'
        self.METRIC_CONDITION = ErrorRate_T1.__name__.lower()
        self.TO_TENSORBOARD = True
        self.VALIDATION = True
        self.PRETRAINED = False
        self.DEBUG = False
        self.SHUFFLE = True
        self.DEVICE = 'cuda'


class HyperParameters:
    def __init__(self, att, b, num_classes):
        self.NUM_EPOCHS = 100
        self.LEARNING_RATE = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.lr_case = 'ImageNet'
        self.IN_CHANNELS = 3
        self.NUM_CLASSES = num_classes
        self.train_batch_size = 32
        self.test_batch_size = 32
        self.resize_shape = 256
        self.input_shape = 224
        self.BACKBONE = b
        self.ATTENTION = att
        self.METRIC_CONDITION = 'min'
        self.TRANSFORM = False



class MAIN:
    def __init__(self, fname, att, b, experiment):
        self.params = Parameters(fname, experiment=experiment)
        self.hyperparams = HyperParameters(att, b, num_classes=10 if experiment == 'CIFAR10/' else 1000)

        self.model = ResNet(backbone_model=self.hyperparams.BACKBONE,
                            in_channels=self.hyperparams.IN_CHANNELS,
                            out_channels=self.hyperparams.NUM_CLASSES,
                            attention=self.hyperparams.ATTENTION)

        self.metrics = [ErrorRate_T1(),
                        ErrorRate_T5(k=5)
                        ]

        self.criterion = CELoss(reduction='mean')

        self.optimizer = SGD(params= self.model.parameters(),
                             lr=self.hyperparams.LEARNING_RATE,
                             momentum=self.hyperparams.momentum,
                             weight_decay=self.hyperparams.weight_decay)

        if self.params.experiment == 'ImageNet/':


            self.transforms = Transforms(resize=self.hyperparams.resize_shape,
                                         shape=self.hyperparams.input_shape,
                                         transform=self.hyperparams.TRANSFORM)


            self.dataset = ImageNetDataset(train_image_dir=self.params.train_data_dir,
                                           train_label_dir=self.params.train_label_dir,
                                           val_image_dir=self.params.val_data_dir,
                                           val_label_dir=self.params.val_label_dir,
                                           train_transforms=self.transforms.train_transforms,
                                           val_transforms=self.transforms.val_transforms,
                                           debug=self.params.DEBUG)

        elif self.params.experiment == 'CIFAR10/' or 'CIFAR100/':

            self.transforms = CIFARTransforms()
            self.dataset = CIFARDataset(path=self.params.path,
                                        train_transform=self.transforms.train_transforms,
                                        test_transform=self.transforms.val_transforms,
                                        dataset=self.params.experiment)


        else:
            raise AttributeError

        self.writer = TensorboardWriter(self.params.LOGDIR)


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
        print(f'BACKBONE : {self.hyperparams.BACKBONE.upper()} ')
        print(f'ATTENTION : {self.hyperparams.ATTENTION} ')
        print(f'Dataset : {self.params.experiment} ')
        print(f'DEVICE : {self.params.DEVICE.upper()} ')


    def run(self):
        self.trainer.fit()

    def validate(self):
        results = self.trainer.validate()
        return results


if __name__ == '__main__':
    experiments = ['ImageNet/']
    # backbone = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101']
    # attention = ['', 'SE', 'CBAM', 'A2', 'RecSE']
    # experiments = ['CIFAR100/']
    backbone = ['ResNet-50']
    attention = ['', 'SE', 'CBAM', 'RecSE']
    # attention = ['SE']


    for exp in experiments:
        for b in backbone:
            for att in attention:
                fname = 'v2_' + b  + att + '/'
                trainer = MAIN(fname=fname, att=att, b=b, experiment=exp)
                trainer.run()
