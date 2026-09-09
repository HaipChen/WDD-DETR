import torch.amp

import math
import sys
from typing import Iterable

import torch
import torch.amp
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import configs
from ..data import CocoEvaluator
from ..misc import dist_utils
from ..optim import ModelEMA, Warmup


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()

    total_epochs = kwargs.get('total_epochs', getattr(configs, 'total_epochs', 200))
    empty_cache_freq = kwargs.get('empty_cache_freq', getattr(configs, 'empty_cache_freq', 100))
    print_freq = kwargs.get('print_freq', getattr(configs, 'print_freq', 500))

    if dist_utils.is_main_process():
        pbar = tqdm(total=len(data_loader), desc=f'Epoch {epoch}/{total_epochs}',
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    else:
        pbar = None

    total_loss = 0
    total_box_loss = 0
    batch_count = 0

    writer :SummaryWriter = kwargs.get('writer', None)
    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        loss_value_scalar = loss_value.detach().item()

        if not math.isfinite(loss_value_scalar):
            print("Loss is {}, stopping training".format(loss_value_scalar))
            print(loss_dict_reduced)
            sys.exit(1)

        total_loss += loss_value.item()

        box_loss = loss_dict_reduced.get('loss_bbox', 0) or loss_dict_reduced.get('bbox', 0)
        if box_loss:
            total_box_loss += box_loss.item()
        batch_count += 1

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        else:
            gpu_mem = 0

        avg_loss = total_loss / batch_count
        avg_box_loss = total_box_loss / batch_count if total_box_loss > 0 else 0

        instances = 0
        for target in targets:
            if 'labels' in target:
                instances += len(target['labels'])
            elif 'boxes' in target:
                instances += len(target['boxes'])

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                'GPU_mem': f'{gpu_mem}MB',
                'loss': f'{avg_loss:.2f}',
                'box_loss': f'{avg_box_loss:.3f}',
                'instances': f'{instances}',
                'e_freq': empty_cache_freq,
                'p_freq': print_freq,
                'tot_epochs': total_epochs
            })

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        if i % empty_cache_freq == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if pbar is not None:
        pbar.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.barrier()

    return {
        'loss': avg_loss,
        'loss_bbox': avg_box_loss,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()


    if dist_utils.is_main_process():
        pbar = tqdm(data_loader, desc='Test', bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    else:
        pbar = None

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types

    for samples, targets in (pbar if pbar is not None else data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


        if pbar is not None:
            pbar.set_postfix({
                'AP': 'calculating...',
                'AR': 'calculating...'
            })

    if pbar is not None:
        pbar.close()

    # gather the stats from all processes
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator