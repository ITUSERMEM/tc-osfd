import os
import sys
import math
import itertools
from collections import defaultdict

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
    CFG, FFT, GPU, BATCH_SIZE, LR, MMD_WEIGHT,
    set_seed, load_target_openset, extract_all_features, eval_tam_ccat_from_features
)
from pilot_class_conditional_mmd import enumerate_tasks, evaluate_known_accuracy, train_one_model
from pilot_idea_inference_variants import eval_per_class_threshold


SEEDS = [8, 9, 10]
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'notebook_classcond')


def train_one_model_target_finetune(model, ld_ttr, optimizer, ce_criterion, device,
                                    epochs=5, entropy_weight=0.1):
    """Fine-tune a pretrained model on target-train known data with CE + entropy minimization."""
    model.train()
    itt = iter(ld_ttr)
    steps = epochs * (len(ld_ttr.dataset) // ld_ttr.batch_size)
    for i in range(1, steps + 1):
        try:
            xt, yt = next(itt)
        except StopIteration:
            itt = iter(ld_ttr)
            xt, yt = next(itt)
        xt, yt = xt.to(device), yt.to(device)

        logits, _ = model(xt)
        ce_loss = ce_criterion(logits, yt)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        loss = ce_loss + entropy_weight * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'  [ft] step {i}/{steps} ce={ce_loss.item():.4f} ent={entropy.item():.4f} '
                  f'total={loss.item():.4f}', flush=True)


def main():
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    kw = {'num_workers': 0, 'pin_memory': False}
    project_root = os.path.dirname(os.path.dirname(__file__))
    root_path = os.path.join(project_root, CFG['data_root'])
    tasks = enumerate_tasks(CFG['loads'])
    target_tasks = [t for t in tasks if t['tgt'] == 9]

    results = []
    for seed in SEEDS:
        set_seed(seed)
        print(f'\n=== SEED {seed} ===', flush=True)
        for task in target_tasks:
            s_loads = task['src']
            tgt_load = task['tgt']
            task_idx = tasks.index(task)
            print(f'\n--- Task {task_idx}: {s_loads} -> {tgt_load} ---', flush=True)

            ld_a = load_source_domain(
                root_path, f'load{s_loads[0]}_train', FFT, CFG['class_num'],
                CFG['samples_per_class'], CFG['src_type_a'], BATCH_SIZE, kw)
            ld_b = load_source_domain(
                root_path, f'load{s_loads[1]}_train', FFT, CFG['class_num'],
                CFG['samples_per_class'], CFG['src_type_b'], BATCH_SIZE, kw)
            ld_ttr, ld_tts = load_target_openset(root_path, tgt_load, FFT, CFG, BATCH_SIZE, kw)

            num_known = CFG['num_known']
            cc_path = os.path.join(SAVE_DIR, f'pu_task{task_idx}_seed{seed}_classcond.pth')

            # Load or train CC-MMD model
            if os.path.exists(cc_path):
                print('Loading existing class-conditional MMD checkpoint...', flush=True)
                ckpt = torch.load(cc_path, map_location=device)
                model = CNN_1D_ResNet18(num_classes=num_known).to(device)
                model.load_state_dict(ckpt['model'])
            else:
                print('Training class-conditional MMD model from scratch...', flush=True)
                model = CNN_1D_ResNet18(num_classes=num_known).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
                ce_criterion = nn.CrossEntropyLoss()
                train_one_model(
                    model, ld_a, ld_b, ld_ttr, optimizer, ce_criterion,
                    device, 500, LR, MMD_WEIGHT, num_known,
                    eval_interval=500, use_class_conditional=True)

            # Baseline TAM+CCAT without fine-tuning
            fd = extract_all_features(
                model, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)
            base = eval_tam_ccat_from_features(fd, num_known)
            print(f'  CC-MMD baseline: H={base["H"]:.2f}% known={base["known"]:.2f}% '
                  f'unk_rej={base["unk_rej"]:.2f}% FA={base["FA"]:.2f}%', flush=True)

            # Target-domain fine-tune with CE + entropy minimization
            model_ft = CNN_1D_ResNet18(num_classes=num_known).to(device)
            model_ft.load_state_dict(model.state_dict())
            optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=LR * 0.1, weight_decay=5e-4)
            ce_criterion = nn.CrossEntropyLoss()
            train_one_model_target_finetune(
                model_ft, ld_ttr, optimizer_ft, ce_criterion, device,
                epochs=5, entropy_weight=0.1)

            fd_ft = extract_all_features(
                model_ft, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)
            ft = eval_tam_ccat_from_features(fd_ft, num_known)
            ft_pc = eval_per_class_threshold(fd_ft, num_known)
            ft_pc_cal = eval_per_class_threshold(fd_ft, num_known, calibrate=True)
            print(f'  CC-MMD + target FT: H={ft["H"]:.2f}% known={ft["known"]:.2f}% '
                  f'unk_rej={ft["unk_rej"]:.2f}% FA={ft["FA"]:.2f}%', flush=True)
            print(f'  CC-MMD + target FT + per-class: H={ft_pc["H"]:.2f}% known={ft_pc["known"]:.2f}% '
                  f'unk_rej={ft_pc["unk_rej"]:.2f}% FA={ft_pc["FA"]:.2f}%', flush=True)
            print(f'  CC-MMD + target FT + per-class+cal: H={ft_pc_cal["H"]:.2f}% known={ft_pc_cal["known"]:.2f}% '
                  f'unk_rej={ft_pc_cal["unk_rej"]:.2f}% FA={ft_pc_cal["FA"]:.2f}%', flush=True)

            results.append({
                'seed': seed,
                'task': task_idx,
                'src': f'{s_loads[0]}+{s_loads[1]}',
                'tgt': tgt_load,
                'base_H': base['H'],
                'ft_H': ft['H'],
                'ft_pc_H': ft_pc['H'],
                'ft_pc_cal_H': ft_pc_cal['H'],
            })

    print('\n=== Target fine-tuning pilot summary ===', flush=True)
    print('seed task src->tgt base_H   ft_H ft_pc_H ft_pc_cal_H', flush=True)
    for r in results:
        print(f'{r["seed"]} {r["task"]:4d} {r["src"]:>7s}->{r["tgt"]} '
              f'{r["base_H"]:6.2f} {r["ft_H"]:6.2f} {r["ft_pc_H"]:7.2f} {r["ft_pc_cal_H"]:11.2f}', flush=True)
    if results:
        base_list = [r['base_H'] for r in results]
        ft_list = [r['ft_H'] for r in results]
        ft_pc_list = [r['ft_pc_H'] for r in results]
        ft_pc_cal_list = [r['ft_pc_cal_H'] for r in results]
        print(f'Baseline avg={np.mean(base_list):.2f}% min={np.min(base_list):.2f}%', flush=True)
        print(f'Finetune avg={np.mean(ft_list):.2f}% min={np.min(ft_list):.2f}%', flush=True)
        print(f'Finetune+perclass avg={np.mean(ft_pc_list):.2f}% min={np.min(ft_pc_list):.2f}%', flush=True)
        print(f'Finetune+perclass+cal avg={np.mean(ft_pc_cal_list):.2f}% min={np.min(ft_pc_cal_list):.2f}%', flush=True)


if __name__ == '__main__':
    main()
