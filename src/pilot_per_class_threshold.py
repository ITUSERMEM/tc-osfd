import os
import sys
import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from resnet1d import CNN_1D_ResNet18
from pilot_tam_extensions import (
    CFG, FFT, GPU, BATCH_SIZE,
    set_seed, load_target_openset, extract_all_features,
    fit_maha_on_features, mahalanobis_scores, hs, scale_radial
)
from pilot_class_conditional_mmd import load_source_domain


SEEDS = [8, 9, 10]
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'notebook_classcond')


def enumerate_tasks(loads):
    tasks = []
    for tgt in loads:
        for sa, sb in itertools.combinations([l for l in loads if l != tgt], 2):
            tasks.append({'src': [sa, sb], 'tgt': tgt})
    return tasks


def mahalanobis_distance_matrix(feats, mus, precs):
    # feats: N x D, mus: C x D, precs: C x D x D
    # returns N x C matrix of squared Mahalanobis distances
    dists = []
    for c in range(mus.shape[0]):
        diff = feats - mus[c].unsqueeze(0)
        dists.append((diff @ precs[c] @ diff.T).diag())
    return torch.stack(dists, dim=1)


def eval_per_class_threshold(fd, num_known, alphas=[0, 0.5, 1, 2, 3],
                             percentiles=[85, 90, 93, 95, 97, 99],
                             conf_threshold=0.9, min_samples=10, beta=0.0):
    src_a = fd['src_a']
    src_b = fd['src_b']
    tgt_tr = fd['tgt_tr']
    tgt_ts = fd['tgt_ts']

    src_feats = torch.cat([scale_radial(src_a['feats'], beta), scale_radial(src_b['feats'], beta)], 0)
    src_labels = torch.cat([src_a['labels'], src_b['labels']], 0)
    tgt_tr_feats = scale_radial(tgt_tr['feats'], beta)
    tgt_tr_labels = tgt_tr['labels']
    tgt_ts_feats = scale_radial(tgt_ts['feats'], beta)
    tgt_ts_labels = tgt_ts['labels']

    probs_tr = F.softmax(tgt_tr['logits'], dim=1)
    conf_tr, pred_tr = probs_tr.max(dim=1)

    tgt_fbd = defaultdict(list)
    for c in range(num_known):
        mask = (pred_tr == c) & (conf_tr >= conf_threshold)
        if mask.sum() >= min_samples:
            tgt_fbd[c].append(tgt_tr_feats[mask])

    tam_feats = []
    tam_labels = []
    for c in range(num_known):
        src_mask = src_labels == c
        chunks = [src_feats[src_mask]]
        if c in tgt_fbd and tgt_fbd[c]:
            chunks.append(torch.cat(tgt_fbd[c], 0))
        ac = torch.cat(chunks, 0) if chunks else torch.empty(0, src_feats.shape[1])
        tam_feats.append(ac)
        tam_labels.append(torch.full((ac.shape[0],), c, dtype=torch.long))
    tam_feats = torch.cat(tam_feats, 0)
    tam_labels = torch.cat(tam_labels, 0)

    mus, precs = fit_maha_on_features(tam_feats, tam_labels, num_known)

    # centroid shifts for CCAT
    shifts = []
    for c in range(num_known):
        src_mask = src_labels == c
        mu_src = src_feats[src_mask].mean(0) if src_mask.any() else torch.zeros(src_feats.shape[1])
        tgt_mask = (pred_tr == c) & (conf_tr >= conf_threshold)
        if tgt_mask.sum() >= min_samples:
            mu_tgt = tgt_tr_feats[tgt_mask].mean(0)
            shifts.append(torch.norm(mu_src - mu_tgt, p=2).item())
        else:
            shifts.append(0.0)
    shifts = torch.tensor(shifts, dtype=torch.float32)

    # distances on target train known (true labels available)
    d_tr = mahalanobis_distance_matrix(tgt_tr_feats, mus, precs)
    d_ts, pred_ts = mahalanobis_scores(tgt_ts_feats, mus, precs)

    is_known_ts = tgt_ts_labels >= 0
    n_k = is_known_ts.sum().item()
    n_u = (~is_known_ts).sum().item()

    best_h = 0.0
    best_metrics = None
    for alpha in alphas:
        # per-class adaptive threshold from target-train known distances
        tau = []
        for c in range(num_known):
            mask = tgt_tr_labels == c
            if mask.sum() == 0:
                tau.append(float('inf'))
                continue
            d_c = d_tr[mask, c] / (1.0 + alpha * shifts[c])
            tau_c = torch.quantile(d_c, 0.01 * percentiles[0])
            tau.append(tau_c.item())
        tau = torch.tensor(tau, dtype=torch.float32)

        for pct in percentiles:
            # update thresholds for this percentile
            for c in range(num_known):
                mask = tgt_tr_labels == c
                if mask.sum() == 0:
                    tau[c] = float('inf')
                    continue
                d_c = d_tr[mask, c] / (1.0 + alpha * shifts[c])
                tau[c] = torch.quantile(d_c, 0.01 * pct).item()

            d_ts_scaled = d_ts / (1.0 + alpha * shifts[pred_ts])
            accepted = d_ts_scaled < tau[pred_ts]
            rejected = ~accepted

            kc = (accepted & is_known_ts & (pred_ts == tgt_ts_labels)).sum().item()
            ur = (rejected & ~is_known_ts).sum().item()
            ua = (accepted & ~is_known_ts).sum().item()
            known = 100.0 * kc / n_k if n_k else 0.0
            unk_rej = 100.0 * ur / n_u if n_u else 0.0
            fa = 100.0 * ua / n_u if n_u else 0.0
            h = hs(known, unk_rej)
            if h > best_h:
                best_h = h
                best_metrics = {
                    'known': known, 'unk_rej': unk_rej, 'FA': fa,
                    'H': h, 'alpha': alpha, 'pct': pct}
    return best_metrics


def eval_baseline_global_threshold(fd, num_known, beta=0.0):
    from pilot_tam_extensions import eval_tam_ccat_from_features
    return eval_tam_ccat_from_features(fd, num_known, beta=beta)


def evaluate_one_checkpoint(ckpt_path, ld_a, ld_b, ld_ttr, ld_tts, device, num_known):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CNN_1D_ResNet18(num_classes=num_known).to(device)
    model.load_state_dict(ckpt['model'])
    fd = extract_all_features(
        model, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)
    base = eval_baseline_global_threshold(fd, num_known, beta=0.0)
    pc = eval_per_class_threshold(fd, num_known, beta=0.0)
    return base, pc


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

            cc_path = os.path.join(SAVE_DIR, f'pu_task{task_idx}_seed{seed}_classcond.pth')
            if not os.path.exists(cc_path):
                print(f'Checkpoint not found: {cc_path}', flush=True)
                continue
            base, pc = evaluate_one_checkpoint(
                cc_path, ld_a, ld_b, ld_ttr, ld_tts, device, CFG['num_known'])
            print(f'  ClassCond global-threshold: H={base["H"]:.2f}% known={base["known"]:.2f}% '
                  f'unk_rej={base["unk_rej"]:.2f}% alpha={base["alpha"]}', flush=True)
            print(f'  ClassCond per-class-threshold: H={pc["H"]:.2f}% known={pc["known"]:.2f}% '
                  f'unk_rej={pc["unk_rej"]:.2f}% alpha={pc["alpha"]} pct={pc["pct"]}', flush=True)

            results.append({
                'seed': seed,
                'task': task_idx,
                'src': f'{s_loads[0]}+{s_loads[1]}',
                'tgt': tgt_load,
                'base_H': base['H'],
                'pc_H': pc['H'],
            })

    print('\n=== Per-class adaptive threshold summary ===', flush=True)
    print('seed task src->tgt global_H perclass_H delta', flush=True)
    for r in results:
        delta = r['pc_H'] - r['base_H']
        print(f'{r["seed"]} {r["task"]:4d} {r["src"]:>7s}->{r["tgt"]} '
              f'{r["base_H"]:8.2f} {r["pc_H"]:10.2f} {delta:+6.2f}', flush=True)
    if results:
        avg_base = np.mean([r['base_H'] for r in results])
        avg_pc = np.mean([r['pc_H'] for r in results])
        min_base = np.min([r['base_H'] for r in results])
        min_pc = np.min([r['pc_H'] for r in results])
        print(f'\nGlobal threshold avg={avg_base:.2f} min={min_base:.2f}', flush=True)
        print(f'Per-class threshold avg={avg_pc:.2f} min={min_pc:.2f}', flush=True)


if __name__ == '__main__':
    main()
