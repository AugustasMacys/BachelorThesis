import torch
from torch.optim import lr_scheduler
import logging
import time
import os

from training.trainModel import create_data_loaders, freeze_until

from utilities import MODELS_DIECTORY, HRNET_CONFIG_FILE

from xray.cnnB import get_seg_model
from xray.cnnC import get_nnc


from x_ray_config import config, update_config


log = logging.getLogger(__name__)


MAX_ITERATIONS = 200000
BATCHES_PER_EPOCH = 5000
WARM_UP = 50000

X_RAY_MODEL_SAVE_PATH_B = os.path.join(MODELS_DIECTORY, "x_ray_model_B.pth")
X_RAY_MODEL_SAVE_PATH_C = os.path.join(MODELS_DIECTORY, "x_ray_model_C.pth")


def evaluate_and_test_xray(model_nnb, model_nnc, criterion_nnc, minimum_loss, dataloaders):
    model_nnb.eval()
    model_nnc.eval()
    running_loss = 0

    for images, labels in dataloaders["val"]:
        with torch.no_grad():
            images = images.to(gpu)
            labels = labels.to(gpu)

            face_x_rays = model_nnb(images)
            classifications = model_nnc(face_x_rays)

            loss_nnc = criterion_nnc(classifications, labels)
            running_loss += loss_nnc.item() * images.size(0)

    total_loss = running_loss / dataset_size["val"]
    log.info('Validation Loss: {:4f}'.format(total_loss))

    if total_loss < minimum_loss:
        minimum_loss = total_loss

    return minimum_loss


def train_xray(epochs, scheduler, optimizer, modelb, modelc, dataloaders, criterion_nnb, criterion_nnc):
    since = time.time()
    minimum_loss = 0.69  # loss of guessing of 0.5 to everything
    iteration = 0
    for epoch in range(epochs):
        modelb.train()
        modelc.train()

        if iteration >= MAX_ITERATIONS:
            break

        # if iteration >= WARM_UP:
        #     modelb

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

            real_x_ray = modelb(img_real)
            fake_x_ray = modelb(img_fake)
            prediction_real = modelc(real_x_ray)
            prediction_fake = modelc(fake_x_ray)

            loss_nnb_real = criterion_nnb(real_x_ray, mask_real)
            loss_nnb_fake = criterion_nnb(fake_x_ray, mask_fake)

            loss_nnc_real = criterion_nnc(prediction_real, 0)
            loss_nnc_fake = criterion_nnc(prediction_fake, 1)

            # original paper used 100 but we are more interested in final result so 20
            # divide by 2 because of fake and real
            loss = (20 * (loss_nnb_real + loss_nnb_fake) + (loss_nnc_real + loss_nnc_fake)) / 2

            loss.backward()
            optimizer.step()

            if iteration > 150000:
                scheduler.step()

            total_examples_real += img_real.size(0)
            total_examples_fake += img_fake.size(0)

            curr_loss_fake = 20 * loss_nnb_fake + loss_nnc_fake
            curr_loss_real = 20 * loss_nnb_real + loss_nnc_real

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

        epoch_loss = (running_fake_loss + running_real_loss) / (total_examples_real + total_examples_fake)
        log.info('Training Loss: {:.4f}'.format(epoch_loss))
        history["train"].append(epoch_loss)

        validation_loss = evaluate_and_test_xray(modelb, modelc, criterion_nnc, minimum_loss, dataloaders)
        history["val"].append(validation_loss)
        log.info(history)

        if validation_loss < minimum_loss:
            minimum_loss = validation_loss
            log.info("Minimum loss is: {}".format(minimum_loss))
            torch.save(modelc.state_dict(), X_RAY_MODEL_SAVE_PATH_B)
            torch.save(modelc.state_dict(), X_RAY_MODEL_SAVE_PATH_C)

        log.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        log.info('Loss: {:4f}'.format(minimum_loss))

        return


if __name__ == '__main__':
    m = torch.nn.Linear(20, 30)
    input = torch.randn(128, 20)
    print(input.size())
    output = m(input)
    print(output.size())

    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    update_config(config, HRNET_CONFIG_FILE)

    # segmentation_model = smp.DeepLabV3Plus(
    #     encoder_name="timm-efficientnet-b4",
    #     encoder_depth=3,
    #     encoder_weights="noisy-student",
    #     classes=2,
    #     in_channels=1
    # )

    # input_example = torch.autograd.Variable(torch.randn(batch_size, 1, 224, 224))
    # output = segmentation_model(input_example)
    # # Two because two classes one for black and one for white
    # assert output.size() == torch.Size([batch_size, 2, 224, 224]), "Model outputs incorrect shape"

    model_nnb = get_seg_model(cfg=config)
    model_nnc = get_nnc()

    model_nnb.to(gpu)
    freeze_until(model_nnb, 'transition1.0.0.weight')  # Transfer learning
    model_nnc.to(gpu)

    batch_size = 16
    epochs = 25

    log.info("Models are initialised")

    nnb_parameters_to_optimize = [p for p in model_nnb.parameters() if p.requires_grad]

    train_loader, validation_loader = create_data_loaders(batch_size, num_workers=6, X_RAY=True)

    dataloaders = {
        "train": train_loader,
        "val": validation_loader
    }

    dataset_size = {
        "val": len(validation_loader.dataset)
    }

    log.info(f"Dataloaders Created")

    criterion_nnb = torch.nn.BCEWithLogitsLoss()
    criterion_nnc = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(nnb_parameters_to_optimize + list(model_nnc.parameters()),
                                 lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    history = {
        "train": [],
        "val": []
    }

    log.info(f"Start Training")

    train_xray(epochs, lr_scheduler, optimizer, model_nnb, model_nnc, dataloaders, criterion_nnb, criterion_nnc)
