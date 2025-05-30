import os
import argparse
import logging
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from modules.dataset import MREProcessor, MNRE_MI_Dataset
from modules.trainer import RETrainer
from models.BertModels import GLRA

logging.basicConfig(format = '%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt = '%m/cd/%Y %H:%M:%S',
                    level = logging.INFO)


def set_seed(seed=42):
    """set random seed"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='./dataset')
    parser.add_argument('--image_path',
                        type=str,
                        default='./dataset/images')
    parser.add_argument('--aux_image_path',
                        type=str,
                        default='./dataset/images_vg')
    parser.add_argument('--aux_image_dict',
                        type=str,
                        default='./dataset/images_vg.pth')
    parser.add_argument('--text_model_name',
                        type=str,
                        default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--image_model_name',
                        type=str,
                        default='../pretrained_models/ViTL-16')
    parser.add_argument('--num_epochs',
                        default=8,
                        type=int,
                        help="num training epochs")
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help="cuda or cpu")
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help="batch size")
    parser.add_argument('--lr',
                        default=5e-5,
                        type=float,
                        help="learning rate")
    parser.add_argument('--warmup_ratio',
                        default=0.01,
                        type=float)
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help="random seed, default is 1")
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int)

    args = parser.parse_args()
    
    fileHandler = logging.FileHandler(f'./logs/result.log', mode='a', encoding='utf8')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s -   %(message)s')
    fileHandler.setFormatter(file_format)
    logger = logging.getLogger(__name__)
    logger.addHandler(fileHandler)
    for k,v in vars(args).items():
        logger.info(" " + str(k) +" = %s", str(v))

    set_seed(args.seed)
    
    processor = MREProcessor(args)
    train_dataset = MNRE_MI_Dataset(processor, args, f'train.txt')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dev_dataset = MNRE_MI_Dataset(processor, args, f'val.txt')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset = MNRE_MI_Dataset(processor, args, f'test.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True) 

    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    model = GLRA(num_labels, processor.tokenizer, args)
    trainer = RETrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                    model=model, processor=processor, args=args, logger=logger)
    trainer.train()
    trainer.test()