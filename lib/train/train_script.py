# In train_script.py

import os
# loss function related
from lib.utils.box_ops import giou_loss, iouhead_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss

# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.odtrack import build_odtrack
# forward propagation related
from lib.train.actors import ODTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss
# --- START OF MODIFICATION ---
import lpips
# --- END OF MODIFICATION ---


def run(settings):
    settings.description = 'Training script for ODTrack with Appearance Prediction Network'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "odtrack":
        net = build_odtrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
        
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    
    # Loss functions and Actors
    if settings.script_name == "odtrack":
        focal_loss = FocalLoss()
        # --- START OF MODIFICATION ---
        # Initialize perceptual loss and add new objectives and weights
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.0, 'cls': 1.0}
        
        # Add appearance prediction losses if APN is used
        if getattr(cfg.MODEL, "APN", None) and cfg.MODEL.APN.USE:
            perceptual_loss_fn = lpips.LPIPS(net='vgg').to(settings.device)
            objective['l1_appearance'] = l1_loss
            objective['perceptual'] = perceptual_loss_fn
            loss_weight['l1_appearance'] = cfg.TRAIN.L1_APPEARANCE_WEIGHT
            loss_weight['perceptual'] = cfg.TRAIN.PERCEPTUAL_WEIGHT
        # --- END OF MODIFICATION ---
        actor = ODTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # --- START OF MODIFICATION ---
    # Optimizer, parameters, and learning rates with staged training
    train_stage = getattr(cfg.TRAIN, "TRAIN_STAGE", "e2e")
    
    if train_stage == 'apn_only':
        print("TRAINING STAGE: Appearance Prediction Network (APN) ONLY")
        # Freeze all parameters first
        for name, param in net.named_parameters():
            param.requires_grad = False
        # Unfreeze only the APN parameters
        for name, param in net.named_parameters():
            if "apn" in name:
                param.requires_grad = True
                if settings.local_rank in [-1, 0]:
                    print(f"Unfreezing: {name}")
    elif train_stage == 'e2e':
        print("TRAINING STAGE: End-to-End (E2E)")
        # Ensure all parameters are trainable
        for param in net.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown training stage: {train_stage}. Must be 'apn_only' or 'e2e'.")
        
    param_dicts = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    # --- END OF MODIFICATION ---

    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)