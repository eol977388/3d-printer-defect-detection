# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    """
    YOLOv5 核心训练函数
    流程：加载模型 → 配置优化器/调度器 → 构建数据加载器 → epoch 循环(前向→反向→优化→验证→保存)
    """

    # ==================== 1. 解析参数 ====================
    # 从 opt 对象提取所有训练配置，赋值给局部变量方便使用
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')  # 触发训练前回调（如 WandB 初始化）

    # ==================== 2. 创建输出目录 ====================
    # runs/train/exp/weights/ 下存放 last.pt（每轮更新）和 best.pt（最佳模型）
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # ==================== 3. 加载超参数 ====================
    # 从 hyp.yaml 读取超参数字典（lr、momentum、loss 增益、增强参数等）
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)   # 保存超参数副本，供 resume 使用
        yaml_save(save_dir / 'opt.yaml', vars(opt))  # 保存命令行参数副本

    # ==================== 4. 初始化日志器（WandB/Comet/TensorBoard）====================
    data_dict = None
    if RANK in {-1, 0}:  # 仅主进程
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # ==================== 5. 基本配置 ====================
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)  # 固定随机种子，保证可复现
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # 校验数据集 yaml，返回 train/val 路径、类别数、类名
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # ==================== 6. 加载/创建模型 ====================
    # 两种模式：
    #   (a) 加载预训练权重 .pt → 迁移学习（推荐，收敛快）
    #   (b) 从 cfg.yaml 从头构建 → 无预训练（工业场景偶尔用）
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect cfg与ckpt['model']不一致时，取交集
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP 是否支持混合精度训练

    # ==================== 7. 冻结层（可选）====================
    # freeze=[10] → 冻结骨干网络前 10 层，只训练检测头
    # 用途：best.pt 已 97%，用少量新数据微调时冻结防止破坏已有特征
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # ==================== 8. 图像尺寸校验 ====================
    # 确保输入尺寸是 stride（通常 32）的整数倍，否则下采样时尺寸不匹配
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # ==================== 9. 自动估算 batch_size（单卡可选）====================
    # batch_size=-1 时自动根据显存容量估算最佳 batch_size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # ==================== 10. 优化器 ====================
    # nbs=64 是"标称 batch_size"，实际 batch_size 不足 64 时通过梯度累加模拟
    # 这样不同 batch_size 下学习率行为一致
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # ==================== 11. 学习率调度器 ====================
    # 两种策略：
    #   cos_lr=True  → 余弦退火（lr 从 lr0 平滑降到 lr0*lrf）
    #   cos_lr=False → 线性衰减（lr 从 lr0 线性降到 lr0*lrf）
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # ==================== 12. EMA（指数移动平均）====================
    # 对模型权重做滑动平均，提升泛化性。推理时用 EMA 模型而非原始模型
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # ==================== 13. 断点续训 ====================
    # resume=True 时从 last.pt 恢复：加载优化器状态、EMA、起始 epoch、最佳 fitness
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # ==================== 14. 多卡模式 ====================
    # DP（DataParallel）：简单但效率低，不推荐
    # DDP（DistributedDataParallel）：推荐的多卡方式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm：多卡训练时同步 BN 统计量，提升精度
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # ==================== 15. 训练数据加载器 ====================
    # create_dataloader 做了：读取图片路径 → Mosaic/增强 → 标签分配 → 批量加载
    # augment=True 开启训练增强（Mosaic、HSV、翻转等），val 时 augment=False
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights, #
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    # 汇总所有标签，校验类别下标不越界
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # ==================== 16. 验证数据加载器（仅主进程）====================
    # val 的 batch_size 是 train 的 2 倍（验证不需要反向传播，显存占用少）
    # rect=True 矩形推理（保持原图比例不 padding 到正方形，减少无效计算）
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        # AutoAnchor：自动检查 anchor 是否匹配数据集目标尺寸，不匹配则重新聚类
        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # ==================== 17. 设置模型属性 ====================
    # 根据检测层数量、类别数、图像尺寸缩放 loss 增益，使不同配置下 loss 量级一致
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # 类别数挂到模型上，推理时需要
    model.hyp = hyp  # 超参数挂到模型上，loss 计算时需要
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 类别权重（样本少的类权重大）
    model.names = names

    # ==================== 18. 训练前最后准备 ====================
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # warmup 迭代次数（前几个 epoch 用小学习率热身）
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1  # 上一次执行 optimizer.step 的 batch 序号（用于梯度累加）
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # AMP 混合精度梯度缩放器（防止 FP16 梯度下溢）
    stopper, stop = EarlyStopping(patience=opt.patience), False  # 早停：patience 轮无提升则停止
    compute_loss = ComputeLoss(model)  # 初始化 loss 计算器（含正负样本匹配、GIoU/cls/obj 三项 loss）
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    # ==================== 19. 训练主循环 ====================
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # 图像加权采样（可选）：根据类别权重和 mAP 给图片分配采样概率，少的类采样更多
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses (box, obj, cls)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)  # DDP 模式下确保各卡每个 epoch 数据顺序不同
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()

        # ---------- batch 循环 ----------
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # 从训练开始累计的 batch 序号
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8→float32, 0-255→0-1

            # ---------- 19a. Warmup 热身（前 nw 个 batch）----------
            # 学习率从 0 线性升到 lr0，动量也做插值，防止初始权重直接用大学习率导致发散
            if ni <= nw:
                xi = [0, nw]  # x interp range
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # ---------- 19b. 多尺度训练（可选）----------
            # 随机在 imgsz*0.5 ~ imgsz*1.5 之间选尺寸，增强模型对不同分辨率的鲁棒性
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # ---------- 19c. 前向传播 ----------
            # AMP 混合精度：前向用 FP16 加速，反向自动转 FP32 保精度
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss = box + obj + cls
                if RANK != -1:
                    loss *= WORLD_SIZE  # DDP 梯度平均补偿
                if opt.quad:
                    loss *= 4.

            # ---------- 19d. 反向传播 ----------
            scaler.scale(loss).backward()  # GradScaler 缩放 loss 后反向传播

            # ---------- 19e. 优化器更新（含梯度累加）----------
            # 每累积 accumulate 个 batch 才执行一次 optimizer.step（模拟大 batch_size）
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # 反向缩放梯度到正常范围
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 梯度裁剪，防爆炸
                scaler.step(optimizer)  # optimizer.step（AMP 版）
                scaler.update()  # 更新 scaler 缩放因子
                optimizer.zero_grad()
                if ema:
                    ema.update(model)  # 更新 EMA 权重
                last_opt_step = ni

            # ---------- 19f. 日志打印 ----------
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # 累计平均 loss
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # ---------- 19g. 学习率调度 ----------
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()  # 每个 epoch 结束后更新学习率

        # ---------- 19h. 验证 & 保存（仅主进程）----------
        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 把模型属性同步到 EMA 模型，保证推理时用的 EMA 也有完整属性
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                # 用 EMA 模型在验证集上计算 P/R/mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # 更新最佳 fitness（P/R/mAP@.5/mAP@.5-.95 的加权组合）
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # ---------- 19i. 保存模型 ----------
            # 每个 epoch 保存 last.pt；如果 fitness 创新高同时保存 best.pt
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),  # 模型权重（FP16 节省空间）
                    'ema': deepcopy(ema.ema).half(),  # EMA 权重
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),  # 优化器状态（resume 需要）
                    'opt': vars(opt),  # 命令行参数（resume 需要）
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:  # 定期额外保存
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # ---------- 19j. 早停广播（DDP 模式）----------
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    # ==================== 20. 训练结束后处理 ====================
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # 从 .pt 中剔除优化器状态，减小文件体积（部署时不需要优化器）
                if f is best:
                    # 用 best.pt 做最终验证，打印详细指标 + 生成 PR 曲线等图表
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    """
    argparse 专门来管理参数的库
    default = 填充具体值
    Root是根目录:https://github.com/ultralytics/yolov5/releases/tag/v7.0 下载对应的初始预训练权重.
    
    重要
    """
    # old_path =ROOT / 'yolov5s.pt'
    # path = r"D:\AI_Tutorial_Related\yolov5-7.0\yolov5-7.0\runs\train\exp34\weights\last.pt"
    # path = r"D:\AI_Tutorial_Related\yolov5-7.0\yolov5-7.0\runs\train\exp2\weights\last.pt"
    path = r""
    parser.add_argument('--weights', type=str, default=path, help='initial weights path')
    """
    模型配置文件:填充yaml文件
    
    weights和cfg规则：
    1. 当weights 不为空时，cfg为空，则加载weights中的模型。开放域的时候用. 大部分时间用。
        --weights, default = ".pt"
        --cfg, default = ""
        
    2. 当weights为空，cfg不为空时，则加载cfg中的模型。但时此时无预训练权重。 工业域时用。
        --weights, default = ""
        --cfg, default = ".yaml"
        
    3. 当weights与cfg模型的结构一样时，就是给cfg赋初始weights。 跟步骤1相同。
        --weights, default = ".pt"  
        --cfg, default = ".yaml"
    
    4. 当weights与cfg模型不一样时，取交集赋weights . 第一个版本训练m模型。m模型效果不错。
        --weights, default = "m.pt"  
        --cfg, default = "n.yaml"  可以做到在上一轮最好的weights的基础，继续修改模型。
        
        m的权重，部分迁移到n中。n就有一个好的出身。
        
        **** 能够继续大模型的初始权重
        **** 什么时候用初始权重，什么时候不用初始权重？
        开放域：生活场景，用
        工业场景：可以尝试先用。也可以尝试不用。
        
    """
    model_path = r"models\yolov5s.yaml"
    parser.add_argument('--cfg', type=str, default=model_path, help='model.yaml path')

    """配置训练数据源 重要"""
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')

    """训练参数数文件地址 重要"""
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low_1.yaml', help='hyperparameters path')

    # 训练epoch次数 epoch=50, 100, 300 重要。 注意学习率，数据量，与epoch的关系，以及训练时间。
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')

    # batch 8，16，32，64 重要
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')

    """
        训练的输入尺寸 640,输出头20,40,80
        1280，输出头40,80, 160， 如果此时，开了多尺度，输出头可能会动态变成320。 r可能高，误检也可能高。
        1120, 35,70,140
        重要
        
    """
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    # 矩形训练。
    parser.add_argument('--rect', action='store_true', help='rectangular training')

    r""" 是否在上一轮的基础上继续训练 重要 
    上一轮中断结束时，最后的权重。需要配合weights参数
    path = r"D:\AI_Tutorial_Related\yolov5-7.0\yolov5-7.0\runs\train\exp4\weights\last.pt"  放在weights的参数位置 
    default = True
    
    如果开启resume时，修改.py中的参数是无效的。同时修改hyp.scratch-low.yaml文件也是无效的。
    
    run\exp\hyp.yaml 或者run\exp\opt.yaml
    """
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')

    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')

    # 训练后有验证。关闭验证
    parser.add_argument('--noval', action='store_true', default=False, help='only validate final epoch')

    # 不进行自动聚类anchor。只有自己聚类的时候，才default = True
    parser.add_argument('--noautoanchor', action='store_true', default=False, help='disable AutoAnchor')

    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')

    """
    重要参数：起到的作用类似分类中的class_weights。过采样。对少的类别进行倾斜。
    
    """
    parser.add_argument('--image-weights', action='store_true',default=True,  help='use weighted image selection for training')
    """
    0, 1, 2 代表用3块显卡
    """
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    """
    是否开启多尺度训练
    
    640标准
    最小320,最大960
    0.5, 1.0, 1.5  重要
    """
    parser.add_argument('--multi-scale', action='store_true', default=True, help='vary img-size +/- 50%%')

    """
    是否开启单类训练
    比如80个类别。开启就只会当1个类别。
    """
    parser.add_argument('--single-cls', action='store_true', default=False, help='train multi-class data as single-class')

    """
    优化器的选择 ，重要参数
    """
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='optimizer')

    """
    多卡训练才开启
    """
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')

    """
    多进程读数据。重要开启。4,6,8之间 ，重要参数
    """
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')

    """
    是否开启cos退火
    """
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')

    """
    标签平滑: 降低GT的标准。给标签一个怀疑尺度。实验角度来说，毫无意义。
    """
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')

    """
    早停轮次。如果100轮，val集毫无进展，就早停。重要参数.
    """
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')

    """
    冻结训练: 0代表不冻结。10代表冻结前10层
    1, 3, 5代表冻结下标为1，3，5的层
    
    有针对性的训练。
    微调：best.pt 已经97%。 再来10张漏检，整个模型全部重新训练有风险，影响best.pt。此时，冻结10层，针对此数据集来微调。 重要参数
    
    [3, ] 只冻3
    [3, 4,5] 冻3,4,5
    [3]，前3层
    [p1,p2,p3,p4,p5]
    """
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=42, help='Global training seed') # 幸运数

    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """程序入口：处理 resume 特殊逻辑 → 选择设备 → 调用 train()"""
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()
        # check_requirements()

    # ==================== Resume 特殊处理 ====================
    # resume 时从 opt.yaml 恢复原始全部参数（此时命令行参数被忽略）
    # 仅保留 weights=last.pt 和 resume=True，确保完全接续上次训练
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        # ==================== 正常训练：校验文件路径 ====================
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 自动递增 exp/exp2/exp3...

    # ==================== 设备选择 ====================
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:  # DDP 多卡初始化
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # ==================== 超参数进化（可选，--evolve N）====================
    # 自动搜索最优超参数：跑 N 代，每代用上一代最优结果变异出新的超参数组合
    # 类似遗传算法：选择（fitness 加权）→ 变异（随机扰动）→ 训练 → 评估 → 重复
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    """以函数方式调用训练（import train; train.run(data='coco128.yaml', weights='yolov5s.pt')）"""
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()   # 解析命令行参数
    main(opt)           # 进入主流程
