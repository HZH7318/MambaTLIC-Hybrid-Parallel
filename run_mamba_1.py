# @Date     : 2025/09/17
# Author   : HZH
# File     : run_mamba_1.py
# Description :  对应mamba1版本1_917

from dataloader.dataset import BaseDataset
import os
from torch.utils.data import DataLoader
import torch
from torch import optim
import os
import glob
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader.dataset import BaseDataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config',default="/data/huzhenghao/ml/h/config/run_config/landsat8_train_loss_mamba/run_mamba_v1/16384W.ini", type=str, 
                    help='config.ini')
args = parser.parse_args()

# -------------------------------------------------------------------
from  model.mamba.mambamodel_1_917  import HNet as model
from model.mamba.mambamodel_1_917 import RateDistortion as Loss


# config
from config.config_help_5 import init_config as init
CONFIG,logger,LOG_ROOT = init(args.config)
# -------------------------------------------------------------------

train_config = CONFIG['trainconfig']
test_config = CONFIG['testconfig']

global_config  =  CONFIG['globalconfig']
PRECISION = global_config.getint('precision', 4)
torch.manual_seed(train_config.getint('seed'))
DATAROOT = train_config['dateset_path']
BATCH_SIZE = train_config.getint('batch_size')
VALROOT = train_config['train_val_path']
VAL_BATCH_SIZE = train_config.getint('train_val_batch_size')
MIN_LR = train_config.getfloat('min_lr')
NUM_WORKERS = train_config.getint('num_workers')
step_size = train_config.getint('step_size')
gamma = train_config.getfloat('gamma')
max_epoch = train_config.getint('train_max_epoch')
MODE = global_config.get('mode')

base_channel = train_config.getint('base_channel')
use_clip_max_norm = train_config.getboolean('use_clip_max_norm')

image_data_pixel_rang  = global_config.getfloat('image_data_pixel_rang')
numch_n  = global_config.getint('numch_n')
numch_m  = global_config.getint('numch_m')
clip_max_norm = train_config.getfloat('clip_max_norm')
LRate = train_config.getfloat('lr')
info = train_config['info']
tmuxh = train_config['tmuxh']
lmbda = train_config.getint('lmbda')

pid = os.getpid()
if MODE =="train":
    DEVICE = train_config.get('device')
    VALROOT = train_config['train_val_path']

elif MODE =="test":
    DEVICE = test_config.get('device')
    VALROOT = test_config["test_path"]
logger.info(f"进程号是: {pid} ,{info}")
logger.info(f"tmuxh = {tmuxh} , lmbda = {lmbda},  DEVICE  = {DEVICE}  , clip_max_norm= {clip_max_norm} ")


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MainInfo():
    def __init__(self):


        train_dataset = BaseDataset(
            root_dir=DATAROOT, split='', transform=None)
        
        val_dataset = BaseDataset(
            root_dir=VALROOT, split='')
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        self.val_loader = DataLoader(
            val_dataset, batch_size=VAL_BATCH_SIZE, num_workers=NUM_WORKERS)
        self.start_epoch  = 0
        self.best_psnr  =0
        self.best_ms_ssim =0
        self.max_bpp =0
        self.best_model_info = ""

        self.criterion =  Loss(lmbda)

        net = model(numch_n,numch_m,base_channel)
        
        self.calculate_parameters(net, 1)
        self.net = net.to(DEVICE)

        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=LRate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)
        
        # 除训练以外的模式 需要加载模型
        if train_config.getboolean('continue_train') or MODE =="test":
            self.load_latest_model(net)
       

    def get_info(self):
        return self.train_loader, self.val_loader, self.net, self.criterion, self.optimizer, self.lr_scheduler,self.start_epoch , self.best_psnr ,self.best_ms_ssim,self.max_bpp,self.best_model_info

    def get_test_info(self):
        return self.test_loader, self.net

    def calculate_parameters(self, model, requires_grad: bool):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)
        print(f"Number of parameters (requires_grad={requires_grad}): {params}")

    def load_latest_model(self, net):
        if MODE =="test":
           
            model_file = glob.glob(f'{test_config["test_model_path"]}/bestmodel.pth')
            if not model_file:
                raise FileNotFoundError("No .pth model found in pretrained_path.")
            
            logger.info(f"TEST-model--->{model_file}")
            
            logger.info(f'Loading checkpoint: {model_file}')

            checkpoint = torch.load(model_file[0])
            
            net.load_state_dict(checkpoint['model'])
        
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_psnr = checkpoint.get('best_psnr', 0)
            self.best_ms_ssim = checkpoint.get('best_ms_ssim', 0)
            self.max_bpp = checkpoint.get('max_bpp', 0)
            self.best_model_info = checkpoint.get('best_model_info', 0)


        else:

            
            list_of_files = glob.glob(f'{train_config["pretrained_path"]}/*.pth')
            if not list_of_files:
                raise FileNotFoundError("No .pth model found in pretrained_path.")
            
            latest_file = max(list_of_files, key=os.path.getctime)
            
            logger.info(f'Loading checkpoint: {latest_file}')

            checkpoint = torch.load(latest_file)
            
            net.load_state_dict(checkpoint['model'])
        
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_psnr = checkpoint.get('best_psnr', 0)
            self.best_ms_ssim = checkpoint.get('best_ms_ssim', 0)
            self.max_bpp = checkpoint.get('max_bpp', 0)
            self.best_model_info = checkpoint.get('best_model_info', 0)


def process_img_batch(img_batch):
    img_ =  img_batch / image_data_pixel_rang
    return img_

def get_network_output(net, img_bit):
    
    result = net(img_bit)
    return result

def get_all_loss(loss):
    psnr = -10 * torch.log10(loss["mse_loss"])
    ms_ssim_db = -10 * torch.log10(1 - loss["ms_ssim"])
    allloss =  {
        "loss":loss["loss"].item(),
        "imgbpp": loss["img_bpp"].item(),
        "mse": loss["mse_loss"].item(),
        "psnr":psnr.item(),
        "y_bpp": loss["y_bpp"],
        "z_bpp": loss["z_bpp"],
        "ms_ssim": ms_ssim_db.item(),
    }
    return allloss

def get_mean_loss(loss_list,psnr,ms_ssim,bpp,y_bpp,z_bpp,mse):
   
    return  np.mean(loss_list), np.mean(psnr), np.mean(ms_ssim),np.mean(bpp),  np.mean(y_bpp), np.mean(z_bpp), np.mean(mse),



def train_batch(net, optimizer, loss_f, batch, count):
    img_batch, _ = batch
    img_batch = img_batch.float().to(DEVICE)
    img_batch = process_img_batch(img_batch)
    
    B, C, H, W = img_batch.size()

    optimizer.zero_grad()  # 确保清空上一次梯度

    result= get_network_output(net, img_batch)

    loss = loss_f(result,img_batch)

    allloss = get_all_loss(loss)

    loss["loss"].backward()

 
    # def clip_gradient(optimizer, grad_clip):
    #     for group in optimizer.param_groups:
    #         for param in group["params"]:
    #             if param.grad is not None:
    #                 param.grad.data.clamp_(-grad_clip, grad_clip)
    # clip_gradient(optimizer, 5)

    if use_clip_max_norm:
        torch.nn.utils.clip_grad_norm_(net.parameters(),  clip_max_norm)   
    optimizer.step()

    return allloss

def train(main_info: MainInfo):
    train_loader, val_loader, net, loss_f, optimizer, lr_scheduler ,start_epoch , best_psnr ,best_ms_ssim,max_bpp,best_model_info = main_info.get_info()
    log_template_d = f"Epoch {{}}: LR = {{:.{PRECISION}f}}  ,Train_loss = {{:.{PRECISION}f}},Train_PSNR = {{:.{PRECISION}f}}, Train_MS_SSIM = {{:.{PRECISION}f}} ,  Train_BPP = {{:.{PRECISION}f}} , Train_y_bpp = {{:.{PRECISION}f}}  , Train_z_bpp  = {{:.{PRECISION}f}} ,  Train_mse = {{:.{PRECISION}f}}  "
    log_template_v = f" |||  Val_LOSS = {{:.{PRECISION}f}}, Val_PSNR = {{:.{PRECISION}f}}, Val_MS_SSIM = {{:.{PRECISION}f}}   ,  Val_BPP = {{:.{PRECISION}f}} , Val_y_bpp = {{:.{PRECISION}f}}  , Val_z_bpp  = {{:.{PRECISION}f}}  ,  Val_mse = {{:.{PRECISION}f}}"

    best_model_info  = ""

    for epoch in range(start_epoch, max_epoch):
        print(f"epoch = {epoch}")
        epoch_count = epoch*len(train_loader)
        net.train()

        loss_list = AverageMeter()
        train_bpp =  AverageMeter()
        train_y_bpp =  AverageMeter()
        train_z_bpp = AverageMeter()

        train_psnr = AverageMeter()
        
        train_ms_ssim = AverageMeter()
        train_mse = AverageMeter()

        with tqdm(total=len(train_loader)) as tbar :
        
            for i, batch in enumerate(train_loader):
                loss = train_batch(net, optimizer, loss_f, batch, epoch_count + i)
                loss_list.update(loss["loss"])
                train_psnr.update(loss["psnr"])
                train_bpp.update(loss["imgbpp"])
                train_y_bpp.update(loss["y_bpp"])
                train_z_bpp.update(loss["z_bpp"])
                train_mse.update(loss["mse"])
                train_ms_ssim.update(loss["ms_ssim"])
                tbar.set_description('Train loss: %.6f  PSNR: %.6f MSSSIM: %.6f bpp: %.6f' % (loss_list.avg, train_psnr.avg,    train_ms_ssim.avg,  train_bpp.avg ))
                tbar.update()
        

        train_loss, train_psnr, train_ms_ssim ,train_bpp, train_y_bpp,train_z_bpp,train_mse  = loss_list.avg,train_psnr.avg,train_ms_ssim.avg,train_bpp.avg, train_y_bpp.avg, train_z_bpp.avg  ,train_mse.avg
         
        
        val_loss, val_psnr ,val_ms_ssim, val_bpp , val_y_bpp,  val_z_bpp  ,val_mse,= val(net, val_loader, epoch)
        current_lr = optimizer.param_groups[0]['lr']

        #                                        学习率        train_loss      train_psnr   train_ms_ssim train_bpp     train_y_bpp    train_z_bpp    train_mse
        log_message_d = log_template_d.format(epoch, current_lr,   train_loss,    train_psnr,  train_ms_ssim , train_bpp,   train_y_bpp,   train_z_bpp  ,train_mse)
        log_message_v = ""
                                           # val_loss   val_psnr   val_ms_ssim  val_bpp  val_y_bpp  val_z_bpp   val_mse
        log_message_v= log_template_v.format( val_loss,    val_psnr,  val_ms_ssim, val_bpp, val_y_bpp, val_z_bpp,  val_mse  )
         
        

        if  val_psnr > best_psnr:
            max_bpp = val_bpp
            best_psnr = val_psnr
            best_ms_ssim = val_ms_ssim
            best_model_info = f'best model :: best_psnr  = {best_psnr},  best_ms_ssim = {best_ms_ssim} , max_bpp = {max_bpp}   || {LOG_ROOT}/{epoch}.pth  '
            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ms_ssim':best_ms_ssim,
                'max_bpp': max_bpp,
                ' best_model_info':best_model_info 
            },  f'{LOG_ROOT}/bestmodel.pth')    
        elif epoch % 5 == 0 :
            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ms_ssim':best_ms_ssim,
                'max_bpp': max_bpp,
                ' best_model_info':best_model_info 
            },  f'{LOG_ROOT}/{epoch}.pth')    
        logger.info(log_message_d  + log_message_v )

        lr_scheduler.step()

        if current_lr < MIN_LR:
            current_lr = MIN_LR
            optimizer.param_groups[0]['lr'] = current_lr
    

    
    logger.info("****************end*************")
    logger.info(best_model_info)




@torch.no_grad()
def val_batch(net, batch, epoch):
    img_batch, _ = batch
    B, C, H, W = img_batch.size()
    img_batch = img_batch.float().to(DEVICE)
    img_batch = process_img_batch(img_batch)
    result_= get_network_output(net, img_batch)
    l = Loss(lmbda)
    result = l(result_,img_batch)
    
    return get_all_loss(result)


@torch.no_grad()
def val(net, val_loader, epoch):
    net.eval()
    loss_list = AverageMeter()
    val_bpp =  AverageMeter()
    val_y_bpp =  AverageMeter()
    val_z_bpp =  AverageMeter()
    val_psnr = AverageMeter()
    val_ms_ssim = AverageMeter()
    val_mse = AverageMeter()

    with tqdm(total=len(val_loader)) as tbar: 

        for i, batch in enumerate(val_loader):
            loss = val_batch(net, batch, epoch)
            loss_list.update(loss["loss"])
            val_psnr.update(loss["psnr"])
            val_ms_ssim.update(loss["ms_ssim"])
            val_bpp.update(loss["imgbpp"])
            
            val_y_bpp.update(loss["y_bpp"])
            val_z_bpp.update(loss["z_bpp"])
            val_mse.update(loss["mse"])

            tbar.set_description('Val loss: %.6f  PSNR: %.6f MSSSIM: %.6f bpp: %.6f' % (loss_list.avg, val_psnr.avg, 
                                                                        val_ms_ssim.avg,val_bpp.avg))
            tbar.update()

    return loss_list.avg,  val_psnr.avg,  val_ms_ssim.avg,  val_bpp.avg, val_y_bpp.avg,  val_z_bpp.avg,  val_mse.avg


@torch.no_grad()
def test(main_info: MainInfo):
    _, val_loader, net, loss_f, _, _ ,start_epoch , best_psnr ,best_ms_ssim,max_bpp,_ = main_info.get_info()
    log_template_d = f" Test_loss = {{:.{PRECISION}f}},Test_PSNR = {{:.{PRECISION}f}}, Test_MS_SSIM = {{:.{PRECISION}f}} ,  Test_BPP = {{:.{PRECISION}f}} , Test_y_bpp = {{:.{PRECISION}f}}  , Test_z_bpp  = {{:.{PRECISION}f}} ,  Test_mse = {{:.{PRECISION}f}}  "


    net.eval()
    loss_list =  AverageMeter()
    test_bpp =   AverageMeter()
    test_y_bpp =   AverageMeter()
    test_z_bpp =   AverageMeter()
    test_psnr =  AverageMeter()
    test_ms_ssim = AverageMeter()
    test_mse = AverageMeter()

    with tqdm(total=len(val_loader)) as tbar :
        
        for i, batch in enumerate(val_loader):
            img_batch, _ = batch
            img_batch = img_batch.float().to(DEVICE)
            img_batch = process_img_batch(img_batch)
            result_= get_network_output(net, img_batch)
      
            result = loss_f(result_,img_batch)
            loss = get_all_loss(result)
            loss_list.update(loss["loss"])
            test_psnr.update(loss["psnr"])
            test_bpp.update(loss["imgbpp"])
            test_y_bpp.update(loss["y_bpp"])
            test_z_bpp.update(loss["z_bpp"])
            test_mse.update(loss["mse"])
            test_ms_ssim.update(loss["ms_ssim"])
            tbar.set_description('Train loss: %.6f  PSNR: %.6f MSSSIM: %.6f bpp: %.6f' % (loss_list.avg, test_psnr.avg,    test_ms_ssim.avg,  test_bpp.avg ))
            tbar.update()
                                           
        log_message= log_template_d.format( loss_list.avg,   test_psnr.avg,  test_ms_ssim.avg, test_bpp.avg, test_y_bpp.avg, test_z_bpp.avg,  test_mse.avg  )
         
        logger.info(log_message)

    logger.info("****************end*************")




def main():
    main_info = MainInfo()
    if MODE == 'train':
        train(main_info)
    elif MODE == 'test':
        test(main_info)
        
if __name__ == "__main__":
    main()
