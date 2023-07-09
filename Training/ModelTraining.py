import copy

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import math
from Utils.Losses import FocalLoss

from Utils.Losses import AdversarialLoss
from Utils.Losses import KDLoss
from Utils.Functions import BalanceCompute
from ProjectDataset.GTA5 import Gta5Dataset
from torch.utils import data
import numpy as np


def TrainSourceOnlyModel(model, train_dataset, num_classes, num_epochs, gpu_dev_id, trained_resume_flag, resumed_epoch):

    learning_rate = 2.5e-4
    momentum = 0.9
    weight_decay = 0.0005

    dont_care_pixel = 255
    loss_at_end_of_epoch = 0

    #Create Seg Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)

    #Create Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

    criterion_seg = FocalLoss(num_class=num_classes,
                              ignore_label=dont_care_pixel)

    training_loader_steps = iter(train_dataset)
    validation_loader_steps = iter(train_dataset)


    epoch_start = 0
    if trained_resume_flag == True:
        epoch_start = resumed_epoch

    with open("source_only_train_log_{}.txt".format(epoch_start), "w") as file:
        file.write("Source Only Train Loss Log.\n")

    if trained_resume_flag == True:
        model_state = torch.load('PreTrainedModels/SourceOnly/deeplab_model_dict_epoch{}.pth'.format(resumed_epoch-1))
        model.load_state_dict(model_state)
        optimizer_state = torch.load('PreTrainedModels/SourceOnly/source_only_optimizer_epoch{}.pth'.format(resumed_epoch-1))
        optimizer.load_state_dict(optimizer_state)
        scheduler_state = torch.load('CityScapesDeeplabModel/source_only_scheduler_epoch{}.pth'.format(resumed_epoch-1))
        scheduler.load_state_dict(scheduler_state)

    for epoch in range(epoch_start, num_epochs):
        model.train()

        for i, data in tqdm(enumerate(train_dataset)):
            inputs, labels, _, __ = data
            inputs, labels = inputs.to(gpu_dev_id), labels.to(gpu_dev_id)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)

            # compute loss
            loss = criterion_seg(outputs, labels)
            loss_at_end_of_epoch = loss

            # backward and optimize
            loss.backward()
            optimizer.step()
            print('Iteration in Epoch{}/{},  Loss : {}'.format(epoch, i, loss_at_end_of_epoch))

        # step scheduler
        scheduler.step()

        print('Epoch {}/{}  Loss : {}'.format(epoch, num_epochs - 1, loss_at_end_of_epoch))
        print('-' * 10)

        torch.save(model.state_dict(), 'PreTrainedModels/SourceOnly/deeplab_model_dict_epoch{}.pth'.format(epoch))
        torch.save(optimizer.state_dict(), 'PreTrainedModels/SourceOnly/source_only_optimizer_epoch{}.pth'.format(epoch))
        torch.save(scheduler.state_dict(), 'PreTrainedModels/SourceOnly/source_only_scheduler_epoch{}.pth'.format(epoch))
        with open("source_only_train_log_{}.txt".format(epoch_start), "a") as file:
            file.write('Iteration in Epoch {},  Loss : {}\n'.format(i, loss_at_end_of_epoch))


def AdvTrainingPixDA(baseline_model,
                  pixelwise_disc,
                  imagewise_disc,
                  source_dataloader,
                  target_dataloader,
                  num_epochs,
                  num_classes,
                  gpu_worker_ids,
                  root_dir,
                  batch_size,
                  target_train_loader,
                  resume_training_enable,
                  resume_epoch,
                  num_shots):

    dont_care_pixel = 255
    adv_los_lambda = 0.1


    generator = baseline_model  # pre-trained model
    pixelwise_discriminator = pixelwise_disc
    imagewise_discriminator = imagewise_disc

    criterion_segmentation= FocalLoss(ignore_index=dont_care_pixel)
    criterion_Adv= AdversarialLoss(gpu=gpu_worker_ids)
    criterion_discriminators = nn.BCEWithLogitsLoss()

    # SGD for the generator (DeepLabV2 model)
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.000025, momentum=0.9, weight_decay=0.0005)
    # Adam for the discriminators
    optimizer_D_pixel = torch.optim.Adam(pixelwise_discriminator.parameters(), lr=0.00001, betas=(0.9, 0.99))
    optimizer_D_image = torch.optim.Adam(imagewise_discriminator.parameters(), lr=0.00001, betas=(0.9, 0.99))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)
    lr_scheduler_D_pixel = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)
    lr_scheduler_D_image = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                             lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

    sample_selection_inc = 0.4
    sample_selection_threshold = 0.4

    start_epoch = 0

    # Training Start
    source_dataset_iteration = iter(source_dataloader)
    target_dataset_iteration = iter(target_dataloader)

    for epoch in range(start_epoch, num_epochs):

        generator.train()
        pixelwise_discriminator.train()
        imagewise_discriminator.train()

        optimizer_G.zero_grad()
        optimizer_D_pixel.zero_grad()
        optimizer_D_image.zero_grad()

        try:
            source_img, source_label, _, _ = next(source_dataset_iteration)
        except:
            source_dataloader,sample_selection_threshold = PerformSampleSelection(segmentation_model=generator,
                                                                                  imagewise_discriminator=imagewise_discriminator,
                                                                                   gpu_id=gpu_worker_ids,
                                                                                   batch_size=batch_size,
                                                                                  root_dir=root_dir,
                                                                                   threshold=sample_selection_threshold,
                                                                                   inc=sample_selection_inc,
                                                                                   source_data_loader= source_dataloader,
                                                                                   target_data_loader=target_dataloader)

            if source_dataloader is not None:
                source_dataset_iteration = iter(source_dataloader)
                source_img, source_label, _, _ = next(source_dataset_iteration)
            else:
                break

        try:
            target_img, target_label, _, _ = next(target_dataset_iteration)
        except:
            target_dataset_iteration = iter(target_dataloader)
            target_img, target_label, _, _ = next(target_dataset_iteration)

        gta5_images = source_img.to(gpu_worker_ids,dtype=torch.float32)
        gta5_labels = source_label.to(gpu_worker_ids, dtype=torch.long)
        cityscapes_images = target_img.to(gpu_worker_ids,dtype=torch.float32)
        cityscapes_labels = target_label.to(gpu_worker_ids, dtype=torch.long)

        ##############################
        # Train the generator (DeepLabv2) and discriminators
        ##############################
        #SourcePrediction
        generated_source_images = generator(gta5_images)
        seg_loss_source = criterion_segmentation(generated_source_images, gta5_labels)
        seg_loss_source.backward(retain_graph=True)

        #TargetPrediction
        generated_target_images = generator(cityscapes_images)
        seg_loss_target = criterion_segmentation(generated_source_images, cityscapes_labels)

        # Forward pass through the discriminator with generated target images
        d_output = pixelwise_discriminator(generated_target_images).to(gpu_worker_ids, dtype=torch.float)
        balanced_target = BalanceCompute(cityscapes_labels)
        loss_pixadv = criterion_Adv(generated_target_images, d_output, cityscapes_labels, balanced_target)
        loss_target = loss_pixadv * adv_los_lambda + seg_loss_target
        loss_target.backward()

        #Train PixelWise Disc
        # No Gradient Operation
        pred_source = generated_source_images.detach()
        pred_target = generated_target_images.detach()

        target_to_source_output = pixelwise_discriminator(pred_source)
        target_to_source_tensor = torch.zeros_like(target_to_source_output).to(gpu_worker_ids, dtype=torch.float)
        target_to_source_tensor.requires_grad_(True)
        source_pixel_wise_loss = criterion_discriminators(target_to_source_output, target_to_source_tensor.to(gpu_worker_ids, dtype=torch.float)) / 2
        source_pixel_wise_loss.backward()

        target_output_discriminator = pixelwise_discriminator(pred_target)
        target_output_tensor = torch.ones_like(target_output_discriminator).to(gpu_worker_ids, dtype=torch.float)
        target_output_tensor.requires_grad_(True)
        target_pixel_wise_loss = criterion_discriminators(target_output_discriminator, target_output_tensor.to(gpu_worker_ids, dtype=torch.float)) / 2
        target_pixel_wise_loss.backward()

        #Train ImageWise Disc
        # No Gradient Operation

        image_wise_source_output = imagewise_discriminator(pred_source)
        image_wise_source_output_tensor = torch.zeros_like(image_wise_source_output).to(gpu_worker_ids, dtype=torch.float)
        image_wise_source_output_tensor.requires_grad_(True)
        image_wise_source_output_loss = criterion_discriminators(image_wise_source_output, image_wise_source_output_tensor.to(gpu_worker_ids, dtype=torch.float)) / 2
        image_wise_source_output_loss.backward()

        image_wise_target_output = imagewise_discriminator(pred_target)
        image_wise_target_output_tensor = torch.ones_like(image_wise_target_output).to(gpu_worker_ids, dtype=torch.float)
        image_wise_target_output_tensor.requires_grad_(True)
        image_wise_target_output_loss = criterion_discriminators(image_wise_target_output, image_wise_target_output_tensor.to(gpu_worker_ids, dtype=torch.float)) / 2
        image_wise_target_output_loss.backward()

        #Train ImageWise Disc For Sample Selection

        optimizer_G.step()
        optimizer_D_pixel.step()
        optimizer_D_image.step()

        lr_scheduler_G.step()
        lr_scheduler_D_pixel.step()
        lr_scheduler_D_image.step()

        print(
            f"Epoch {epoch + 1}: Generator Source Loss: {seg_loss_target.item():.4f}, Generator Target Loss: {seg_loss_target.item():.4f}, Adv Loss: {loss_target.item():.4f}, Disc Source Loss: {source_pixel_wise_loss.item():.4f}, Disc Target Loss: {target_pixel_wise_loss.item():.4f}")

        if epoch % 200 == 1:
          torch.save(generator.state_dict(), 'PreTrainedModels/PixAdv/generator_dict_{}.pth'.format(epoch))
          torch.save(pixelwise_discriminator.state_dict(), 'PreTrainedModels/PixAdv/pixwise_model_dict_{}.pth'.format(epoch))
          torch.save(imagewise_discriminator.state_dict(), 'PreTrainedModels/PixAdv/imagewise_model_dict_{}.pth'.format(epoch))

          torch.save(optimizer_G.state_dict(), 'PreTrainedModels/PixAdv/optimizer_G_dict_{}.pth'.format(num_shots,epoch))
          torch.save(optimizer_D_pixel.state_dict(), 'PreTrainedModels/PixAdv/optimizer_D_pixel_dict_{}.pth'.format(num_shots,epoch))
          torch.save(optimizer_D_image.state_dict(), 'PreTrainedModels/PixAdv/optimizer_D_image_dict_{}.pth'.format(num_shots, epoch))

          torch.save(lr_scheduler_G.state_dict(), 'PreTrainedModels/PixAdv/lr_scheduler_G_dict_{}.pth'.format(epoch))
          torch.save(lr_scheduler_D_pixel.state_dict(), 'PreTrainedModels/PixAdv/lr_scheduler_D_pixel_dict_{}.pth'.format(epoch))
          torch.save(lr_scheduler_D_image.state_dict(), 'PreTrainedModels/PixAdv/lr_scheduler_D_image_dict_{}.pth'.format(epoch))

    kd_epochs = 200
    KnowledgeDistillationMethod(model=generator,
                           num_class=num_classes,
                           gpu_id=gpu_worker_ids,
                           train_dataloader=target_train_loader,
                           num_epochs=kd_epochs,
                            num_shot=num_shots)




def PerformSampleSelection(segmentation_model, imagewise_discriminator, source_data_loader, target_data_loader, gpu_id, threshold, inc, root_dir, batch_size):
    source_img_list = []
    segmentation_model.eval()
    imagewise_discriminator.eval()

    criterion = nn.BCEWithLogitsLoss()


    target_data_iter = iter(target_data_loader)
    with torch.no_grad():
        for index, batch in tqdm(enumerate(source_data_loader)):

            try:
                target_img, _, _, _ = next(target_data_loader)
            except:
                target_data_iter = iter(target_data_loader)
                target_img, _, _, _ = next(target_data_iter)

            source_img, _, _, source_name = batch

            images = source_img.to(gpu_id, dtype=torch.float32)
            output = segmentation_model(images)

            src_img_pred = output
            src_img_pred = src_img_pred.detach()
            disc_ratio = imagewise_discriminator(src_img_pred)

            for id, output in enumerate(disc_ratio):
                target_tensor = torch.zeros_like(output).to(gpu_id, dtype=torch.float)
                loss = criterion(output, target_tensor.to(gpu_id, dtype=torch.float))
                if loss.item() > threshold:
                    source_img_list.append(source_name[id])


    del source_data_loader
    gta5_dataset = Gta5Dataset(root_dir,image_list=source_img_list)
    source_dataset_loader = data.DataLoader(gta5_dataset,
                                            batch_size=batch_size,
                                            num_workers=gpu_id,
                                            drop_last=True,
                                            shuffle=True,
                                            pin_memory=True)


    segmentation_model.train()
    imagewise_discriminator.train()
    incremented_threshold = threshold * inc

    return source_dataset_loader, incremented_threshold


def KnowledgeDistillationMethod(model, train_dataloader, num_class, num_epochs, gpu_id, num_shot=1):

    # SGD for the generator (DeepLabV2 model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000025, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

    # Sample selection parameters

    knowledge_dist_lambda = 0.5
    segmentation_loss = FocalLoss(num_class=num_class)
    kd_loss = KDLoss()
    model_frozen = copy.deepcopy(model)

    train_data_iterator = iter(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        try:
            images, labels, _, _ = next(train_data_iterator)
        except:
            train_data_iterator = iter(train_dataloader)
            images, labels, _, _ = next(train_data_iterator)

        images, labels = images.to(gpu_id, dtype=torch.float32), labels.to(gpu_id, dtype=torch.long)


        with torch.no_grad():
            prediction_frozen = model_frozen(images)

        current_prediction = model(images)

        seg_loss_val = segmentation_loss(current_prediction, labels) + \
                       knowledge_dist_lambda * kd_loss(current_prediction,prediction_frozen, labels)

        seg_loss_val.backward()

        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), 'PreTrainedModels/KD/KD_{}shot.pth'.format(num_shot))



def ValidateModel(model, val_loader, gpu_id):

    model.eval()

    num_classes = 19
    classes_name_19 = {0: "Road", 1: "Sidewalk", 2: "Building", 3: "Wall", 4: "Fence", 5: "Pole", 6: "TLight",7: "TSign",
                       8: "Vegetation",9: "Terrain",10: "Sky",11: "Person",12: "Rider",13: "Car", 14: "Truck",15: "Bus", 16: "Train",17: "Motorcycle",18: "Bicycle"}
    confusion_matrix = np.zeros((num_classes, num_classes))
    total_samples = 0
    Epsilon = 1e-9

    with torch.no_grad():
        for index, batch in tqdm(enumerate(val_loader), disable=False):
            image, label, _, name = batch
            pred_high = model(image.to(gpu_id, dtype=torch.float32))

            interp = torch.nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True).to(gpu_id)
            output = interp(pred_high)

            _, output = output.max(dim=1)
            output = output.cpu().numpy()
            label = label.cpu().numpy()

            for target, preicison in zip(label, output):
                target = target.flatten()
                preicison = preicison.flatten()

                mask = (target >= 0) & (target < num_classes)
                hist = np.bincount(
                    num_classes * target[mask].astype(int) + preicison[mask],
                    minlength=num_classes ** 2,
                ).reshape(num_classes, num_classes)

                confusion_matrix += hist
            total_samples += len(label)


        output_histogram = confusion_matrix

        output_hist_sum = output_histogram.sum(axis=1)
        mask = (output_hist_sum != 0)
        diagonal = np.diag(output_histogram)

        avg_acc = diagonal.sum() / output_histogram.sum()
        acc_cls_c = diagonal / (output_hist_sum + Epsilon)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diagonal / (output_histogram.sum(axis=0) + Epsilon)
        precision_cls = np.mean(precision_cls_c)
        iu = diagonal / (output_hist_sum + output_histogram.sum(axis=0) - diagonal + Epsilon)
        # mean_iu = np.mean(iu[mask])
        mean_iu = round(np.nanmean(iu[mask]) * 100, 2)
        freq = output_histogram.sum(axis=1) / output_histogram.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(num_classes), [round(iu[i] * 100, 2) if m else "X" for i, m in enumerate(mask)]))
        return mean_iu, cls_iu, avg_acc



def validate_pixAdv_from_pixda(model, pixAdv_disc, val_loader, metrics, gpu_id):
    model.eval()
    metrics.reset()
    ignore_label = 255

    results = {}
    val_loss = 0.0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(val_loader), disable=False):
            image, label, _, name = batch
            pred_high = model(image.to(gpu_id, dtype=torch.float32))

            interp = torch.nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True).to(gpu_id)
            output = interp(pred_high)


            #for img, lbl, out in zip(interp(image), label, output):
                #visualizer.display_current_results([(img, lbl, out)], val_iter, 'Val')

            _, output = output.max(dim=1)
            output = output.cpu().numpy()
            label = label.cpu().numpy()
            metrics.update(label, output)

        #visualizer.info(f'Validation loss at iter {val_iter}: {val_loss/len(val_loader)}')
        #visualizer.add_scalar('Validation_Loss', val_loss/len(val_loader), val_iter)
        score = metrics.get_results()
        #visualizer.add_figure("Val_Confusion_Matrix_Recall", score['Confusion Matrix'], step=val_iter)
        #visualizer.add_figure("Val_Confusion_Matrix_Precision", score["Confusion Matrix Pred"], step=val_iter)
        results["Val_IoU"] = score['Class IoU']
        #visualizer.add_results(results)
        #visualizer.add_scalar('Validation_mIoU', score['Mean IoU'], val_iter)
        #visualizer.info(metrics.to_str_print(score))
