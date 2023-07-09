
from ProjectDataset.GTA5 import Gta5Dataset
from ProjectModel import DeepLabv2Resnet101Model
from torch.utils import data
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
from Training.ModelTraining import TrainSourceOnlyModel
from Training.ModelTraining import AdvTrainingPixDA
from ProjectDataset.CityScapes import CityScapesDataset
from Training.Discriminators import ImagewiseDiscriminator
from Training.Discriminators import PixelwiseDiscriminator
from Training.ModelTraining import ValidateModel
import sys

if __name__ == '__main__':
    #configuartion
    print('Your Command Prompt : ', sys.argv)
    test_or_train = 'test'
    test_model = 'KD_5shot'


    if(len(sys.argv) > 1):
        arguments  = sys.argv[1].split(" ")
        test_or_train = arguments[0]
        if test_or_train == 'test':
            test_model = arguments[1]

    training_conf = 'pixel_adversarial'
    root_dir = os.getcwd()
    gpu_id = 0
    train_batch = 2
    num_classes = 19
    adv_num_epochs = 250000
    source_only_num_epochs = 202
    num_shot = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device activated : ',device)

    gta5_dataset = Gta5Dataset(root_dir)
    baseline_model = DeepLabv2Resnet101Model.DeepLabV2(num_classes = num_classes)

    pixelwise_discriminator = PixelwiseDiscriminator()
    imagewise_discriminator = ImagewiseDiscriminator()

    # Move baseline_model to cuda
    baseline_model = nn.DataParallel(baseline_model)
    baseline_model = baseline_model.to(gpu_id)

    # Move discriminators to cuda
    pixelwise_discriminator = nn.DataParallel(pixelwise_discriminator)
    pixelwise_discriminator = pixelwise_discriminator.to(gpu_id)

    imagewise_discriminator = nn.DataParallel(imagewise_discriminator)
    imagewise_discriminator = imagewise_discriminator.to(gpu_id)

    # Set cudnn
    cudnn.benchmark = True
    cudnn.enabled = True

    print('Models created')

    train_dataset_loader = None
    val_dataset_loader = None

    val_cityscapes_dataset = CityScapesDataset(root_dir, target=False, train=False)
    test_dataset_loader = data.DataLoader(val_cityscapes_dataset,
                                          batch_size=train_batch,
                                          num_workers=gpu_id,
                                          drop_last=True,
                                          shuffle=True,
                                          pin_memory=True)

    if test_or_train == 'train':
        if training_conf == 'source_only':

            pixelwise_discriminator = nn.DataParallel(pixelwise_discriminator)
            pixelwise_discriminator = pixelwise_discriminator.to(gpu_id)

            imagewise_discriminator = nn.DataParallel(imagewise_discriminator)
            imagewise_discriminator = imagewise_discriminator.to(gpu_id)

            train_dataset_loader = data.DataLoader(gta5_dataset,
                                                    batch_size=train_batch,
                                                    num_workers=gpu_id,
                                                    drop_last=True,
                                                    shuffle=True,
                                                    pin_memory=True)


            source_only_train = False
            if source_only_train == True:

                trained_resume_flag = True
                resumed_epoch = 0
                if trained_resume_flag == True:
                    resumed_epoch = 0 #Set the last recoreded iteration number + 1

                TrainSourceOnlyModel(baseline_model,
                                     train_dataset_loader,
                                     num_classes,
                                     source_only_num_epochs,
                                     gpu_id,
                                     trained_resume_flag=trained_resume_flag,
                                     resumed_epoch=resumed_epoch)



        elif training_conf == 'pixel_adversarial':
            #Get Source Only Model
            baseline_state = torch.load('./PreTrainedModels/OutputModels/source_only.pth')
            baseline_model.load_state_dict(baseline_state)

            pixelwise_discriminator = nn.DataParallel(pixelwise_discriminator)
            pixelwise_discriminator = pixelwise_discriminator.to(gpu_id)

            imagewise_discriminator = nn.DataParallel(imagewise_discriminator)
            imagewise_discriminator = imagewise_discriminator.to(gpu_id)

            source_dataset_loader = data.DataLoader(gta5_dataset,#remove this
                                                    batch_size=train_batch,
                                                    num_workers=gpu_id,
                                                    drop_last=True,
                                                    shuffle=True,
                                                    pin_memory=True)

            target_cityscapes_dataset = CityScapesDataset(root_dir, target= True, train=False, num_shot=num_shot)
            target_cityscapee_dataset_loader = data.DataLoader(target_cityscapes_dataset,
                                                    batch_size=train_batch,
                                                    num_workers=gpu_id,
                                                    drop_last=True,
                                                    shuffle=True,
                                                    pin_memory=True)

            train_cityscapes_dataset = CityScapesDataset(root_dir, target= False, train=True, num_shot=num_shot)
            train_cityscape_dataset_loader = data.DataLoader(train_cityscapes_dataset,
                                                    batch_size=train_batch,
                                                    num_workers=gpu_id,
                                                    drop_last=True,
                                                    shuffle=True,
                                                    pin_memory=True)


            just_validation_enable = False
            if just_validation_enable == False:

                resume_training = False
                resume_epoch = 0
                AdvTrainingPixDA(baseline_model=baseline_model,
                              pixelwise_disc= pixelwise_discriminator,
                              imagewise_disc=imagewise_discriminator,
                              num_epochs=adv_num_epochs,
                              num_classes=num_classes,
                              source_dataloader=source_dataset_loader,
                              target_dataloader= target_cityscapee_dataset_loader,
                              gpu_worker_ids=gpu_id,
                              root_dir=root_dir,
                              batch_size=train_batch,
                              target_train_loader=train_cityscape_dataset_loader,
                              resume_training_enable=resume_training,
                              resume_epoch=resume_epoch,
                              num_shots=num_shot)



    else:
        baseline_state = torch.load('./PreTrainedModels/OutputModels/{}.pth'.format(test_model))
        baseline_model.load_state_dict(baseline_state)
        mean_iu, cls_iu, avg_acc = ValidateModel(baseline_model, test_dataset_loader, gpu_id)
        print('Mean IuO : ', mean_iu, 'Avg Acc : ', avg_acc, 'Class IuO Detail : ', cls_iu)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
