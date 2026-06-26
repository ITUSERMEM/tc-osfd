import os
import sys
import math
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from resnet1d import CNN_1D_ResNet18
from utils import mmd_rbf_noaccelerate, mmd_class_conditional
from incomplete_health_mmd.data_loader import load_source_domain
from pilot_tam_extensions import (
    CFG, SEED, ITERATION, BATCH_SIZE, LR, MMD_WEIGHT, FFT, GPU,
    set_seed, load_target_openset, extract_all_features, eval_tam_ccat_from_features
)


ONLY_LOAD9 = os.environ.get('CCMMD_FULL', '0') != '1'
SEED = int(os.environ.get('CCMMD_SEED', str(SEED)))
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'notebook_classcond')


def enumerate_tasks(loads):
    tasks = []
    for tgt in loads:
        for sa, sb in itertools.combinations([l for l in loads if l != tgt], 2):
            tasks.append({'src': [sa, sb], 'tgt': tgt})
    return tasks


@torch.no_grad()
def evaluate_known_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def train_one_model(model, ld_a, ld_b, ld_ttr, optimizer, ce_criterion,
                    device, iteration, lr_init, mmd_weight, num_known,
                    eval_interval=500, use_class_conditional=False):
    """Iteration-level training with global or class-conditional MMD."""
    model.train()
    ita, itb, itt = iter(ld_a), iter(ld_b), iter(ld_ttr)
    loss_history = {'total': [], 'ce': [], 'mmd': []}
    eval_history = []
    best_val_acc = 0.0
    best_state = None
    best_iter = 0

    for i in range(1, iteration + 1):
        cur_lr = lr_init / math.pow((1 + 10 * (i - 1) / iteration), 0.75)
        optimizer.param_groups[0]['lr'] = cur_lr

        try:
            xa, ya = next(ita)
        except StopIteration:
            ita = iter(ld_a)
            xa, ya = next(ita)
        try:
            xb, yb = next(itb)
        except StopIteration:
            itb = iter(ld_b)
            xb, yb = next(itb)
        try:
            xt, yt = next(itt)
        except StopIteration:
            itt = iter(ld_ttr)
            xt, yt = next(itt)

        xs = torch.cat([xa, xb], 0).to(device)
        ys = torch.cat([ya, yb], 0).to(device)
        xt = xt.to(device)
        yt = yt.to(device)

        ls, fs = model(xs)
        _, ft = model(xt)

        ce_loss = ce_criterion(ls, ys)
        if use_class_conditional:
            mmd_loss = mmd_class_conditional(fs, ft, ys, yt, num_known)
        else:
            mmd_loss = mmd_rbf_noaccelerate(fs, ft)
        loss = ce_loss + mmd_weight * mmd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history['total'].append(loss.item())
        loss_history['ce'].append(ce_loss.item())
        loss_history['mmd'].append(mmd_loss.item())

        if i % eval_interval == 0:
            val_acc = evaluate_known_accuracy(model, ld_ttr, device)
            eval_history.append({'iter': i, 'val_acc': val_acc})
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_iter = i
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'[eval] iter {i}/{iteration} lr={cur_lr:.6f} '
                  f'total={loss.item():.4f} ce={ce_loss.item():.4f} '
                  f'mmd={mmd_loss.item():.4f} val_acc={val_acc:.4f} '
                  f'best={best_val_acc:.4f}@{best_iter}', flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return loss_history, eval_history, best_state


def run_tam_ccat(model, ld_a, ld_b, ld_ttr, ld_tts, device, num_known):
    fd = extract_all_features(
        model, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)
    return eval_tam_ccat_from_features(fd, num_known)


def main():
    set_seed(SEED)
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    kw = {'num_workers': 0, 'pin_memory': False}
    project_root = os.path.dirname(os.path.dirname(__file__))
    root_path = os.path.join(project_root, CFG['data_root'])
    tasks = enumerate_tasks(CFG['loads'])

    target_tasks = [t for t in tasks if t['tgt'] == 9] if ONLY_LOAD9 else tasks
    print(f'Class-conditional MMD pilot on {len(target_tasks)} tasks: device={device}', flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    results = []
    for task in target_tasks:
        s_loads = task['src']
        tgt_load = task['tgt']
        task_idx = tasks.index(task)
        print(f'\n=== Task {task_idx}: {s_loads} -> {tgt_load} ===', flush=True)

        ld_a = load_source_domain(
            root_path, f'load{s_loads[0]}_train', FFT, CFG['class_num'],
            CFG['samples_per_class'], CFG['src_type_a'], BATCH_SIZE, kw)
        ld_b = load_source_domain(
            root_path, f'load{s_loads[1]}_train', FFT, CFG['class_num'],
            CFG['samples_per_class'], CFG['src_type_b'], BATCH_SIZE, kw)
        ld_ttr, ld_tts = load_target_openset(root_path, tgt_load, FFT, CFG, BATCH_SIZE, kw)

        num_known = CFG['num_known']

        # Global MMD baseline
        print('Training global MMD model...', flush=True)
        model_global = CNN_1D_ResNet18(num_classes=num_known).to(device)
        optimizer_global = torch.optim.Adam(model_global.parameters(), lr=LR, weight_decay=5e-4)
        ce_criterion = nn.CrossEntropyLoss()
        _, _, best_global = train_one_model(
            model_global, ld_a, ld_b, ld_ttr, optimizer_global, ce_criterion,
            device, ITERATION, LR, MMD_WEIGHT, num_known,
            eval_interval=500, use_class_conditional=False)
        if best_global is not None:
            torch.save({'model': best_global, 'cfg': CFG},
                       os.path.join(SAVE_DIR, f'pu_task{task_idx}_seed{SEED}_global.pth'))
        metrics_global = run_tam_ccat(model_global, ld_a, ld_b, ld_ttr, ld_tts, device, num_known)
        print(f'  Global MMD + TAM+CCAT: H={metrics_global["H"]:.2f}% '
              f'known={metrics_global["known"]:.2f}% unk_rej={metrics_global["unk_rej"]:.2f}% '
              f'FA={metrics_global["FA"]:.2f}% alpha={metrics_global["alpha"]}', flush=True)

        # Class-conditional MMD
        print('Training class-conditional MMD model...', flush=True)
        model_cc = CNN_1D_ResNet18(num_classes=num_known).to(device)
        optimizer_cc = torch.optim.Adam(model_cc.parameters(), lr=LR, weight_decay=5e-4)
        _, _, best_cc = train_one_model(
            model_cc, ld_a, ld_b, ld_ttr, optimizer_cc, ce_criterion,
            device, ITERATION, LR, MMD_WEIGHT, num_known,
            eval_interval=500, use_class_conditional=True)
        if best_cc is not None:
            torch.save({'model': best_cc, 'cfg': CFG},
                       os.path.join(SAVE_DIR, f'pu_task{task_idx}_seed{SEED}_classcond.pth'))
        metrics_cc = run_tam_ccat(model_cc, ld_a, ld_b, ld_ttr, ld_tts, device, num_known)
        print(f'  ClassCond MMD + TAM+CCAT: H={metrics_cc["H"]:.2f}% '
              f'known={metrics_cc["known"]:.2f}% unk_rej={metrics_cc["unk_rej"]:.2f}% '
              f'FA={metrics_cc["FA"]:.2f}% alpha={metrics_cc["alpha"]}', flush=True)

        results.append({
            'task': task_idx,
            'src': f'{s_loads[0]}+{s_loads[1]}',
            'tgt': tgt_load,
            'global_H': metrics_global['H'],
            'cc_H': metrics_cc['H'],
            'global_known': metrics_global['known'],
            'cc_known': metrics_cc['known'],
            'global_unk_rej': metrics_global['unk_rej'],
            'cc_unk_rej': metrics_cc['unk_rej'],
        })

    print('\n=== Class-conditional MMD summary ===', flush=True)
    print('Task  src->tgt  global_H  classcond_H   Δ', flush=True)
    for r in results:
        delta = r['cc_H'] - r['global_H']
        print(f'{r["task"]:4d}  {r["src"]:>7s}->{r["tgt"]}  {r["global_H"]:8.2f}  '
              f'{r["cc_H"]:11.2f}  {delta:+.2f}', flush=True)
    if results:
        avg_global = np.mean([r['global_H'] for r in results])
        avg_cc = np.mean([r['cc_H'] for r in results])
        print(f'Average global_H={avg_global:.2f} classcond_H={avg_cc:.2f} Δ={avg_cc-avg_global:+.2f}', flush=True)


if __name__ == '__main__':
    main()
