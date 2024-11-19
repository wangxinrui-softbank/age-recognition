import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import argparse
import collections

import numpy as np


import data_loaders
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

from tqdm import tqdm
from trainer import Trainer
from iresnet import iresnet50
from parse_config import ConfigParser
from utils import prepare_device, update_lr_scheduler
from torch.cuda.amp import autocast, GradScaler

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser):
    logger = config.get_logger("train")

    train_loader = data_loaders.TXTDataLoader(config["data_loader"]["args"]["txt_train"],
                                            config["data_loader"]["args"]["batch_size"],
                                            config["data_loader"]["args"]["shuffle"],
                                            config["data_loader"]["args"]["num_workers"])
    
    val_loader = data_loaders.TXTDataLoader(config["data_loader"]["args"]["txt_val"],
                                            config["data_loader"]["args"]["batch_size"],
                                            config["data_loader"]["args"]["shuffle"],
                                            config["data_loader"]["args"]["num_workers"])

    config = update_lr_scheduler(config, len(train_loader))

    # build model architecture, then print to console
    extractor = iresnet50(False, fp16=True, num_features=512)
    weight = torch.load('irse50_jcv.pt')
    extractor.load_state_dict(weight)
    extractor.eval()
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    if config["checkpoint"]:
        logger.info(f'Loading checkpoint: {config["checkpoint"]} ...')
        checkpoint = torch.load(config["checkpoint"])
        state_dict = checkpoint["state_dict"]
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    extractor = extractor.to(device)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj("optimizer", torch.optim, model.parameters())
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    total_idx = 0
    scaler = GradScaler(enabled=True)
    for epoch in range(config["trainer"]["epochs"]):
        
        model.train()
        for batch_idx, (image, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            with autocast(enabled=True):
                image, target = image.to(device), target.to(device)
                optimizer.zero_grad()

                embedding = extractor(image)
                output = model(embedding)
                loss = criterion(output, target.to(torch.float16)/100.0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_idx += 1
                if np.mod(total_idx, 50) == 0:
                    print("Regression, Epoch:{}, Step:{}, Loss:{}".format(epoch, batch_idx, loss.item()))

        torch.save(model.state_dict(), 'checkpoints/regression_{}.pth'.format(str(epoch).zfill(2)))
        eval(model, extractor, val_loader, device)


def eval(model, extractor, val_loader, device):
    model.eval()
    extractor.eval()
    error_list = list()
    for batch in tqdm(val_loader):
        image = batch[0].to(device)
        embedding = extractor(image)
        output = model(embedding)
        output = output.detach().cpu().numpy()
        target = batch[1].numpy()
        error = np.mean(np.abs(target - output))
        error_list.append(error)
    print('Validation Error:', np.mean(error_list))



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="train_regression.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
