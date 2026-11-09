import configparser
import logging
import os
import time
import sys

def init_log(log_root, need_terminal=False):
    if not os.path.exists(log_root):
        os.makedirs(log_root,exist_ok=True)
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    if need_terminal:
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setLevel(logging.INFO)
        terminal_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"))
        logger.addHandler(terminal_handler)
    # # ↑ 控制台输出
    file_handler = logging.FileHandler(log_root+"/exp.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # ↑ 文件输出
    return logger, log_root


def init_train_log(CONFIG, configpath = None, rebegin_logroot = None, ):
    train_config = CONFIG['trainconfig']
    if not rebegin_logroot:
        log_root = os.path.join(train_config.get('train_log'),
                            time.strftime("%Y%m%d_%H%M")+"_"+configpath)
    else :
        log_root = rebegin_logroot
    logger, _ = init_log(log_root)
    # ↑ 文件输出
    return logger, log_root

def init_test_log(CONFIG, configpath = None):
    test_config = CONFIG['testconfig']
    
    log_root = os.path.join(test_config.get('testlog'),
                            time.strftime("%Y%m%d_%H%M")+"_"+configpath)
    
    logger, _ = init_log(log_root)
    # ↑ 文件输出
    return logger, log_root

def init_config(config_path):
    
    # 读取配置文件
    CONFIG = configparser.ConfigParser()
    CONFIG.read(config_path)

    # print(config_path)
    if CONFIG["globalconfig"]["mode"] == 'train':
        if CONFIG["trainconfig"]["continue_train"] == 'True' :
            logger, LOG_ROOT = init_train_log(CONFIG,config_path.split("/")[-1].split(".")[0],CONFIG["trainconfig"]["pretrained_path"])
        else :
            logger, LOG_ROOT = init_train_log(CONFIG,config_path.split("/")[-1].split(".")[0])

    elif CONFIG["globalconfig"]["mode"] == 'test':
        logger, LOG_ROOT = init_test_log(CONFIG,config_path.split("/")[-1].split(".")[0])
        
    elif CONFIG["globalconfig"]["mode"] == 'e' or CONFIG["trainconfig"]["mode"] == 'd':
        logger, LOG_ROOT = init_log(CONFIG["edconfig"]["enc_root"],config_path.split("/")[-1].split(".")[0])

    return CONFIG,logger,LOG_ROOT


