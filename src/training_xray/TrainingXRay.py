import logging
import os
import time
import pickle

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import torch
from torch.optim import lr_scheduler


from src.training.TrainModelFaces2D import create_data_loaders
from src.Utilities import HRNET_CONFIG_FILE, PREVIEW_MODELS, PREVIEW_TEST, VALIDATION_LABELS, VALIDATION_DATAFRAME_PATH
from src.training_xray.CNNB import get_seg_model
from src.training_xray.CNNC import get_nnc
from src.xray_config import config, update_config


log = logging.getLogger(__name__)


MAX_ITERATIONS = 200000
BATCHES_PER_EPOCH = 2000
LOSS_ALPHA = 100
SCHEDULER_STEP = 50000


X_RAY_MODEL_SAVE_PATH_B = os.path.join(PREVIEW_MODELS, "x_ray_model_B")
X_RAY_MODEL_SAVE_PATH_C = os.path.join(PREVIEW_MODELS, "x_ray_model_C")


def evaluate_and_test_xray(model_nnb, model_nnc, criterion_nnc, minimum_loss, dataloaders,
                           plotting=False):
    model_nnb.eval()
    model_nnc.eval()
    running_loss = 0

    # predictions = []

    for i, (images, labels) in enumerate(dataloaders["val"]):
        with torch.no_grad():
            images = images.to(gpu)
            labels = labels.to(gpu)

            face_x_rays = model_nnb(images)
            classifications = model_nnc(face_x_rays)
            probabilities = model_nnc.predict(face_x_rays)
            labels = labels.type_as(classifications)
            face_x_rays = torch.squeeze(face_x_rays)
            if i == 0 or i == 50 or i == 250 and plotting:
                fig, axes = plt.subplots(2, 2, figsize=(15, 6))
                ax1, ax2, ax3, ax4 = axes.flatten()
                example1 = torch.sigmoid(face_x_rays[0])
                example1 = example1.detach().cpu().numpy()
                example2 = torch.sigmoid(face_x_rays[1])
                example2 = example2.detach().cpu().numpy()
                example3 = torch.sigmoid(face_x_rays[2])
                example3 = example3.detach().cpu().numpy()
                example4 = torch.sigmoid(face_x_rays[3])
                example4 = example4.detach().cpu().numpy()
                ax1.imshow(example1, cmap='gray')
                ax2.imshow(example2, cmap='gray')
                ax3.imshow(example3, cmap='gray')
                ax4.imshow(example4, cmap='gray')
                plt.show()

            loss_nnc = criterion_nnc(classifications, labels.unsqueeze(1))
            running_loss += loss_nnc.item() * images.size(0)
            # predictions.append(probabilities.detach().cpu().numpy())

    # flat_predictions = [item for sublist in predictions for item in sublist]
    # with open("predictions_new.txt", "wb") as fp:  # Pickling
    #     pickle.dump(flat_predictions, fp)
    # ap = average_precision_score(ground_truth, flat_predictions)
    # roc = roc_auc_score(ground_truth, flat_predictions)
    # log.info('AP: {:4f}'.format(ap))
    # log.info('ROC: {:4f}'.format(roc))

    total_loss = running_loss / dataset_size["val"]
    log.info('Validation Loss: {:4f}'.format(total_loss))

    if total_loss < minimum_loss:
        minimum_loss = total_loss

    return minimum_loss


def train_xray(epochs, scheduler, optimizer, modelb, modelc, dataloaders,
               criterion_nnb, criterion_nnc, plotting=False):
    since = time.time()
    minimum_loss = 1
    iteration = 0
    for epoch in range(epochs):
        modelb.train()
        modelc.train()

        if iteration >= MAX_ITERATIONS:
            break

        log.info('Epoch {}/{}'.format(epoch, epochs - 1))
        log.info('-' * 10)

        total_examples = 0
        nnb_accumulated = 0
        nnc_accumulated = 0
        for batch_iteration, pair in enumerate(dataloaders["train"]):
            iteration += 1
            img = pair["img"].to(gpu)
            mask = pair["mask"].to(gpu)
            label = pair["label"].to(gpu)

            optimizer.zero_grad()

            x_ray = modelb(img)
            prediction = modelc(x_ray)
            x_ray = torch.squeeze(x_ray)

            # Plotting
            if batch_iteration == 0 and plotting:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
                mask1 = torch.sigmoid(x_ray[0]).detach().cpu().numpy()
                mask2 = torch.sigmoid(x_ray[1]).detach().cpu().numpy()
                ax1.imshow(mask1, cmap='gray')
                ax2.imshow(mask2, cmap='gray')
                plt.show()

            loss_nnb = criterion_nnb(x_ray, mask)
            loss_nnc = criterion_nnc(prediction.squeeze(), label.float())

            loss = (LOSS_ALPHA * loss_nnb + loss_nnc)

            nnb_accumulated += LOSS_ALPHA * (loss_nnb.item() * img.size(0))
            nnc_accumulated += loss_nnc.item() * img.size(0)

            loss.backward()
            optimizer.step()

            if iteration > SCHEDULER_STEP:
                scheduler.step()

            total_examples += img.size(0)

            if batch_iteration % 50 == 0 and batch_iteration != 0:
                log.info("New 50 batches are evaluated")
                log.info("Batch Number: {}".format(batch_iteration))
                time_elapsed = time.time() - since
                log.info('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                log.info("iteration: {}, max_lr: {}".format(batch_iteration, max_lr))
            if batch_iteration >= BATCHES_PER_EPOCH:
                break

        nnb_average = nnb_accumulated / total_examples
        nnc_average = nnc_accumulated / total_examples

        log.info('NNB Average: {:.4f}'.format(nnb_average))
        log.info('NNC Average: {:.4f}'.format(nnc_average))

        epoch_loss = nnb_average + nnc_average
        log.info('Training Loss: {:.4f}'.format(epoch_loss))
        history["train"].append(epoch_loss)

        if iteration >= 200:
            validation_loss = evaluate_and_test_xray(modelb, modelc, criterion_nnc, minimum_loss, dataloaders)
            history["val"].append(validation_loss)
            log.info(history)

            if validation_loss < minimum_loss:
                minimum_loss = validation_loss
                log.info("Minimum loss is: {}".format(minimum_loss))
                path_model_b = X_RAY_MODEL_SAVE_PATH_B + str(epoch) + ".pth"
                path_model_c = X_RAY_MODEL_SAVE_PATH_C + str(epoch) + ".pth"
                torch.save(modelb.state_dict(), path_model_b)
                torch.save(modelc.state_dict(), path_model_c)

    time_elapsed = time.time() - since
    log.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    log.info('Loss: {:4f}'.format(minimum_loss))

    return


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    update_config(config, HRNET_CONFIG_FILE)

    model_nnb = get_seg_model(cfg=config)
    model_nnc = get_nnc()

    model_nnb.to(gpu)
    model_nnc.to(gpu)

    # model_nnb.load_state_dict(torch.load(r"D:\deepfakes\trained_models\modelsXray\preview_newest_x_ray_model_B9.pth"))
    # model_nnc.load_state_dict(torch.load(r"D:\deepfakes\trained_models\modelsXray\preview_newest_x_ray_model_C9.pth"))

    batch_size = 40
    epochs = 1000

    log.info("Models are initialised")

    train_loader, validation_loader = create_data_loaders(batch_size, num_workers=4, X_RAY=True,
                                                          preview=True)

    dataloaders = {
        "train": train_loader,
        "val": validation_loader
    }
    dataset_size = {
        "val": len(validation_loader.dataset)
    }

    log.info(f"Dataloaders Created")

    criterion_nnc = torch.nn.BCEWithLogitsLoss()
    criterion_nnb = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(list(model_nnb.parameters()) + list(model_nnc.parameters()),
                                 lr=0.0002)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)

    history = {
        "train": [],
        "val": []
    }

    log.info(f"Start Training")

    ground_truth = list(pd.read_csv(VALIDATION_DATAFRAME_PATH).label)

    # evaluate_and_test_xray(model_nnb, model_nnc, criterion_nnc, 1, dataloaders)
    train_xray(epochs, lr_scheduler, optimizer, model_nnb, model_nnc, dataloaders, criterion_nnb, criterion_nnc)
