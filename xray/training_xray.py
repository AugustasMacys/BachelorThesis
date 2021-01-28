import segmentation_models_pytorch as smp

import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
import logging
import time
import os

from training.trainModel import create_data_loaders
from training.trainModel import collate_fn

from utilities import MODELS_DIECTORY

from xray.cnnB import get_seg_model
from xray.cnnC import get_nnc


log = logging.getLogger(__name__)


MAX_ITERATIONS = 200000
BATCHES_PER_EPOCH = 5000

X_RAY_MODEL_SAVE_PATH = os.path.join(MODELS_DIECTORY, "x_ray_model.pth")


def evaluate_xray(model, minimum_loss):
    model.eval()
    running_loss = 0

    for val_batch_iteration, pair in enumerate(dataloaders["val"]):
        pass # Need to implement nnc




def train_xray(epochs, scheduler, modelb, modelc, dataloaders):
    since = time.time()
    minimum_loss = 0.69  # loss of guessing of 0.5 to everything
    iteration = 0
    for epoch in range(epochs):
        modelb.train()
        modelc.train()

        if iteration == MAX_ITERATIONS:
            break

        log.info('Epoch {}/{}'.format(epoch, epochs - 1))
        log.info('-' * 10)

        total_examples_real = 0
        total_examples_fake = 0

        running_fake_loss = 0
        running_real_loss = 0

        for batch_iteration, pair in enumerate(dataloaders["train"]):
            iteration += 1

            img_real = pair["real_image"]
            mask_real = pair["real_mask"]

            img_fake = pair["fake_image"]
            mask_fake = pair["fake_mask"]

            img_real = img_real.to(gpu)
            mask_real = mask_real.to(gpu)
            img_fake = img_fake.to(gpu)
            mask_fake = mask_fake.to(gpu)

            optimizer.zero_grad()

            real_x_ray = modelb(img_real.unsqueeze(1))
            fake_x_ray = modelb(img_fake.unsqueeze(1))
            prediction_real = modelc(real_x_ray)
            prediction_fake = modelc(fake_x_ray)
            with torch.no_grad():
                prediction_real = torch.softmax(prediction_real, dim=1)
                prediction_fake = torch.softmax(prediction_fake, dim=1)

            curr_loss_real = criterion(output_real, mask_real)
            curr_loss_fake = criterion(output_fake, mask_fake)
            loss = (curr_loss_real + curr_loss_fake) / 2

            loss.backward()
            optimizer.step()

            total_examples_real += img_real.size(0)
            total_examples_fake += img_fake.size(0)

            running_fake_loss += curr_loss_fake.item() * img_fake.size(0)
            running_real_loss += curr_loss_real.item() * img_real.size(0)

            if batch_iteration % 250 == 0:
                log.info("New 250 batches are evaluated")
                log.info("Batch Number: {}".format(batch_iteration))
                time_elapsed = time.time() - since
                log.info('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                log.info("iteration: {}, max_lr: {}".format(batch_iteration, max_lr))

            if batch_iteration >= BATCHES_PER_EPOCH:
                break

        scheduler.step()

        epoch_loss = (running_fake_loss + running_real_loss) / (total_examples_real + total_examples_fake)
        log.info('Training Loss: {:.4f}'.format(epoch_loss))
        history["train"].append(epoch_loss)

        validation_loss = evaluate_xray(model, minimum_loss)
        history["val"].append(validation_loss)
        log.info(history)

        if validation_loss < minimum_loss:
            minimum_loss = validation_loss
            log.info("Minimum loss is: {}".format(minimum_loss))
            torch.save(model.state_dict(), X_RAY_MODEL_SAVE_PATH)

        log.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        log.info('Loss: {:4f}'.format(minimum_loss))

        return model


if __name__ == '__main__':
    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # segmentation_model = smp.DeepLabV3Plus(
    #     encoder_name="timm-efficientnet-b4",
    #     encoder_depth=3,
    #     encoder_weights="noisy-student",
    #     classes=2,
    #     in_channels=1
    # )

    model_cnnB = get_seg_model(cfg="pass")
    model_nnc = get_nnc(config="pass")

    model_cnnB.to(gpu)
    model_nnc.to(gpu)

    model_cnnB.module.pretrained_grad()

    batch_size = 16
    # input_example = torch.autograd.Variable(torch.randn(batch_size, 1, 224, 224))
    # output = segmentation_model(input_example)
    # # Two because two classes one for black and one for white
    # assert output.size() == torch.Size([batch_size, 2, 224, 224]), "Model outputs incorrect shape"

    log.info("Model is initialised")

    train_loader, validation_loader = create_data_loaders(batch_size, num_workers=6)
    dataloaders = {
        "train": train_loader,
        "val": validation_loader
    }
    log.info(f"Dataloaders Created")

    criterion = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=0.001,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    history = {
        "train": [],
        "val": []
    }

    segmentation_model.to(gpu)