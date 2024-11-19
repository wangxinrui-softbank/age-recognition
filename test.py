import torch
import argparse
import collections
import numpy as np
import model.loss as module_loss
#import model.model as module_arch
import model.metric as module_metric


from tqdm import tqdm
from iresnet import iresnet50
from trainer import Trainer
from parse_config import ConfigParser
from data_loaders import VALDataLoader
from utils import prepare_device, update_lr_scheduler
from model.model import RegressionResMLP
# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

#best for 1109: epoch7, validation erroe 6.93
def main(args):
    extractor = iresnet50(False, fp16=True, num_features=512)
    e_weight = torch.load('irse50_jcv.pt')
    extractor.load_state_dict(e_weight)
    extractor.eval()
    for idx in range(7,8):
        #model = config.init_obj("arch", module_arch)
        model = RegressionResMLP(dropout=0, num_residuals_per_block=2, 
                                num_blocks=4, num_classes=79, num_initial_features=512)
        m_weight = torch.load('ckpt_1109/clf_weighted_{}.pth'.format(str(idx).zfill(2)), map_location='cpu')
        model.load_state_dict(m_weight)
        model.eval()

        #device, device_ids = prepare_device(config["n_gpu"])
        model = model.to(args.device)
        extractor = extractor.to(args.device)
        val_loader = VALDataLoader('testset_crop', 32, False, 8)

        loss_list, error_list = list(), list()
        criterion = torch.nn.CrossEntropyLoss()
        for batch in tqdm(val_loader):
            image = batch[0].to(args.device)
            target = batch[1].to(args.device)
            with torch.no_grad():
                embedding = extractor(image)
                output = model(embedding)
            loss = criterion(output, target)
            loss_list.append(loss.item())

            output = torch.argmax(torch.softmax(output, dim=1), dim=1)
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            error = np.mean(np.abs(target - output))
            error_list.append(error)
        print('Epoch:', idx, 'Validation Loss:', np.mean(loss_list), 'Validation Error:', np.mean(error_list))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default="train_clf.json",
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default='cpu',
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    '''
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    config.device = args.device
    '''
    args = parser.parse_args()
    main(args)
