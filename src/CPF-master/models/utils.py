import yaml
import torch


def get_training_config(config_path, model_name):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = dict(full_config['global'], **full_config[model_name])
    specific_config['model_name'] = model_name
    return specific_config


def get_pre_config(config_path, args):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = dict(full_config[args.ptype][args.teacher][args.dataset])
    return specific_config


def check_device(conf):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(conf['device'])
    if conf['model_name'] in ['DeepWalk', 'GraphSAGE']:
        is_cuda = False
    else:
        is_cuda = not conf['no_cuda'] and torch.cuda.is_available()
    # ----------not set random seed----------
    # if is_cuda:
    #     torch.cuda.manual_seed(conf['seed'])
    #     torch.cuda.manual_seed_all(conf['seed'])  # if you are using multi-GPU.
    if conf['device'] >= 0:
        device = torch.device("cuda:" + str(conf['device'])) if is_cuda else torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    return device
