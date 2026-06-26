import os
import sys
import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from resnet1d import CNN_1D_ResNet18
from pilot_tam_extensions import (
    CFG, FFT, GPU, BATCH_SIZE,
    set_seed, load_target_openset, extract_all_features,
    fit_maha_on_features, mahalanobis_scores, hs
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
    dists = []
    for c in range(mus.shape[0]):
        diff = feats - mus[c].unsqueeze(0)
        dists.append((diff @ precs[c] @ diff.T).diag())
    return torch.stack(dists, dim=1)


def fit_maha_ledoit_wolf(feats, labels, num_classes, reg=1e-4):
    """Fit per-class Gaussian with Ledoit-Wolf shrinkage covariance."""
    fbd = defaultdict(list)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            fbd[c].append(feats[mask])
    mus, precs = [], []
    d = feats.shape[1]
    eye = torch.eye(d)
    for c in range(num_classes):
        if fbd[c]:
            ac = torch.cat(fbd[c], 0).numpy()
            mu = torch.from_numpy(ac.mean(0))
            if ac.shape[0] > 1:
                lw = LedoitWolf().fit(ac)
                cov = torch.from_numpy(lw.covariance_).float() + reg * eye
            else:
                cov = eye / reg
            try:
                prec = torch.linalg.inv(cov)
            except Exception:
                prec = eye / reg
        else:
            mu = torch.zeros(d)
            prec = eye / reg
        mus.append(mu)
        precs.append(prec)
    return torch.stack(mus), torch.stack(precs)


def eval_per_class_threshold(fd, num_known, alphas=[0],
                             percentiles=[80, 85, 90, 93, 95, 97, 99],
                             conf_threshold=0.9, min_samples=10,
                             lw=False, blend=False, calibrate=False):
    """Per-class adaptive threshold with optional Ledoit-Wolf, covariance blending, or feature calibration."""
    src_a = fd['src_a']
    src_b = fd['src_b']
    tgt_tr = fd['tgt_tr']
    tgt_ts = fd['tgt_ts']

    src_feats = torch.cat([src_a['feats'], src_b['feats']], 0)
    src_labels = torch.cat([src_a['labels'], src_b['labels']], 0)
    tgt_tr_feats = tgt_tr['feats']
    tgt_tr_labels = tgt_tr['labels']
    tgt_ts_feats = tgt_ts['feats']
    tgt_ts_labels = tgt_ts['labels']

    # Test-time feature calibration: z-score using target-train known stats per dimension
    if calibrate:
        mu_cal = tgt_tr_feats.mean(0)
        std_cal = tgt_tr_feats.std(0).clamp_min(1e-8)
        tgt_tr_feats = (tgt_tr_feats - mu_cal) / std_cal
        tgt_ts_feats = (tgt_ts_feats - mu_cal) / std_cal
        src_feats = (src_feats - mu_cal) / std_cal

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

    fit_fn = fit_maha_ledoit_wolf if lw else fit_maha_on_features
    mus, precs = fit_fn(tam_feats, tam_labels, num_known)

    # centroid shifts for CCAT scaling
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

    if blend:
        # Fit source-only covariances for blending
        src_mus, src_precs = fit_fn(src_feats, src_labels, num_known)

    d_tr = mahalanobis_distance_matrix(tgt_tr_feats, mus, precs)
    d_ts, pred_ts = mahalanobis_scores(tgt_ts_feats, mus, precs)

    is_known_ts = tgt_ts_labels >= 0
    n_k = is_known_ts.sum().item()
    n_u = (~is_known_ts).sum().item()

    best_h = 0.0
    best_metrics = None
    for alpha in alphas:
        for pct in percentiles:
            tau = torch.full((num_known,), float('inf'), dtype=torch.float32)
            for c in range(num_known):
                mask = tgt_tr_labels == c
                if mask.sum() == 0:
                    continue
                d_c = d_tr[mask, c] / (1.0 + alpha * shifts[c])
                tau[c] = torch.quantile(d_c, 0.01 * pct)

            if blend:
                # Blend target and source Mahalanobis distances per class
                d_ts_src, _ = mahalanobis_scores(tgt_ts_feats, src_mus, src_precs)
                d_ts_eff = 0.5 * d_ts + 0.5 * d_ts_src
            else:
                d_ts_eff = d_ts

            d_ts_scaled = d_ts_eff / (1.0 + alpha * shifts[pred_ts])
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
                    'H': h, 'alpha': alpha, 'pct': pct,
                }
    return best_metrics


def eval_per_class_threshold_detailed(fd, num_known, alphas=[0],
                                      percentiles=[80, 85, 90, 93, 95, 97, 99],
                                      conf_threshold=0.9, min_samples=10):
    """Per-class adaptive threshold with detailed per-class FA/FR breakdown."""
    best_metrics = eval_per_class_threshold(
        fd, num_known, alphas=alphas, percentiles=percentiles,
        conf_threshold=conf_threshold, min_samples=min_samples)
    best_alpha = best_metrics['alpha']
    best_pct = best_metrics['pct']

    src_a = fd['src_a']
    src_b = fd['src_b']
    tgt_tr = fd['tgt_tr']
    tgt_ts = fd['tgt_ts']

    src_feats = torch.cat([src_a['feats'], src_b['feats']], 0)
    src_labels = torch.cat([src_a['labels'], src_b['labels']], 0)
    tgt_tr_feats = tgt_tr['feats']
    tgt_tr_labels = tgt_tr['labels']
    tgt_ts_feats = tgt_ts['feats']
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

    d_tr = mahalanobis_distance_matrix(tgt_tr_feats, mus, precs)
    d_ts, pred_ts = mahalanobis_scores(tgt_ts_feats, mus, precs)

    tau = torch.full((num_known,), float('inf'), dtype=torch.float32)
    for c in range(num_known):
        mask = tgt_tr_labels == c
        if mask.sum() == 0:
            continue
        d_c = d_tr[mask, c] / (1.0 + best_alpha * shifts[c])
        tau[c] = torch.quantile(d_c, 0.01 * best_pct)

    d_ts_scaled = d_ts / (1.0 + best_alpha * shifts[pred_ts])
    accepted = d_ts_scaled < tau[pred_ts]
    rejected = ~accepted

    is_known_ts = tgt_ts_labels >= 0

    per_class = []
    for c in range(num_known):
        known_mask = is_known_ts & (tgt_ts_labels == c)
        n_kc = known_mask.sum().item()
        correct_accept = (accepted & known_mask & (pred_ts == c)).sum().item()
        false_reject = (rejected & known_mask).sum().item()
        misclass_accept = (accepted & known_mask & (pred_ts != c)).sum().item()
        unknown_mask = ~is_known_ts
        false_accept_c = (accepted & unknown_mask & (pred_ts == c)).sum().item()
        per_class.append({
            'class': c,
            'n_known': n_kc,
            'correct_accept': correct_accept,
            'false_reject': false_reject,
            'misclass_accept': misclass_accept,
            'false_accept_unknown': false_accept_c,
            'known_acc': 100.0 * correct_accept / n_kc if n_kc else 0.0,
            'false_reject_rate': 100.0 * false_reject / n_kc if n_kc else 0.0,
        })

    n_u = (~is_known_ts).sum().item()
    total_false_accept = (accepted & ~is_known_ts).sum().item()
    return {
        **best_metrics,
        'per_class': per_class,
        'total_unknown': n_u,
        'total_false_accept': total_false_accept,
        'overall_false_accept_rate': 100.0 * total_false_accept / n_u if n_u else 0.0,
    }


def eval_knn_rejection(fd, num_known, ks=[1, 3, 5],
                       conf_threshold=0.9, min_samples=10):
    """k-NN distance to target-train known samples per class as rejection score."""
    src_a = fd['src_a']
    src_b = fd['src_b']
    tgt_tr = fd['tgt_tr']
    tgt_ts = fd['tgt_ts']

    src_feats = torch.cat([src_a['feats'], src_b['feats']], 0)
    src_labels = torch.cat([src_a['labels'], src_b['labels']], 0)
    tgt_tr_feats = tgt_tr['feats']
    tgt_tr_labels = tgt_tr['labels']
    tgt_ts_feats = tgt_ts['feats']
    tgt_ts_labels = tgt_ts['labels']

    probs_tr = F.softmax(tgt_tr['logits'], dim=1)
    conf_tr, pred_tr = probs_tr.max(dim=1)

    is_known_ts = tgt_ts_labels >= 0
    n_k = is_known_ts.sum().item()
    n_u = (~is_known_ts).sum().item()

    # Source-domain classifier for predicted class
    # k-NN per class using target-train known samples
    best_h = 0.0
    best_metrics = None
    for k in ks:
        # Compute class-wise k-NN distance for target train and test
        dist_tr = torch.full((tgt_tr_feats.shape[0], num_known), float('inf'))
        dist_ts = torch.full((tgt_ts_feats.shape[0], num_known), float('inf'))
        for c in range(num_known):
            mask = tgt_tr_labels == c
            if mask.sum() < k:
                continue
            nn_c = NearestNeighbors(n_neighbors=k, algorithm='auto')
            nn_c.fit(tgt_tr_feats[mask].numpy())
            d_tr_c, _ = nn_c.kneighbors(tgt_tr_feats.numpy())
            d_ts_c, _ = nn_c.kneighbors(tgt_ts_feats.numpy())
            dist_tr[:, c] = torch.from_numpy(d_tr_c.mean(1)).float()
            dist_ts[:, c] = torch.from_numpy(d_ts_c.mean(1)).float()

        pred_ts = dist_ts.argmin(1)
        d_ts_min = dist_ts.min(1).values

        for pct in range(5, 95, 1):
            # Use target-train known distances to set per-class thresholds
            tau = torch.full((num_known,), float('inf'), dtype=torch.float32)
            for c in range(num_known):
                mask = tgt_tr_labels == c
                if mask.sum() == 0:
                    continue
                tau[c] = torch.quantile(dist_tr[mask, c], 0.01 * pct)
            accepted = d_ts_min < tau[pred_ts]
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
                    'H': h, 'k': k, 'pct': pct,
                }
    return best_metrics


def evaluate_one_checkpoint(ckpt_path, ld_a, ld_b, ld_ttr, ld_tts, device, num_known):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CNN_1D_ResNet18(num_classes=num_known).to(device)
    model.load_state_dict(ckpt['model'])
    fd = extract_all_features(
        model, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)

    base = eval_per_class_threshold(fd, num_known)
    lw = eval_per_class_threshold(fd, num_known, lw=True)
    blend = eval_per_class_threshold(fd, num_known, blend=True)
    cal = eval_per_class_threshold(fd, num_known, calibrate=True)
    knn = eval_knn_rejection(fd, num_known)
    return {
        'base': base, 'lw': lw, 'blend': blend, 'cal': cal, 'knn': knn,
    }


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
            res = evaluate_one_checkpoint(
                cc_path, ld_a, ld_b, ld_ttr, ld_tts, device, CFG['num_known'])

            for name, m in res.items():
                print(f'  {name:6s}: H={m["H"]:.2f}% known={m["known"]:.2f}% '
                      f'unk_rej={m["unk_rej"]:.2f}% FA={m["FA"]:.2f}%', flush=True)

            results.append({
                'seed': seed,
                'task': task_idx,
                'src': f'{s_loads[0]}+{s_loads[1]}',
                'tgt': tgt_load,
                **{f'{name}_H': m['H'] for name, m in res.items()},
            })

    print('\n=== Inference-variant pilot summary ===', flush=True)
    print('seed task src->tgt base_H   lw_H blend_H  cal_H  knn_H', flush=True)
    for r in results:
        print(f'{r["seed"]} {r["task"]:4d} {r["src"]:>7s}->{r["tgt"]} '
              f'{r["base_H"]:6.2f} {r["lw_H"]:6.2f} {r["blend_H"]:7.2f} '
              f'{r["cal_H"]:6.2f} {r["knn_H"]:6.2f}', flush=True)

    if results:
        for name in ['base', 'lw', 'blend', 'cal', 'knn']:
            hs_list = [r[f'{name}_H'] for r in results]
            print(f'{name:6s}: avg={np.mean(hs_list):.2f}% min={np.min(hs_list):.2f}%', flush=True)


if __name__ == '__main__':
    main()
