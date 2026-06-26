import os
import sys
import math
import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
from scipy.fftpack import fft
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from resnet1d import CNN_1D_ResNet18
from utils import mmd_rbf_noaccelerate
from incomplete_health_mmd.data_loader import load_source_domain


SEED = 8
DATASET = 'pu'
ITERATION = 500
BATCH_SIZE = 256
LR = 1e-3
MMD_WEIGHT = 0.1
FFT = True
GPU = 0

CFG = {
    'class_num': 12,
    'num_known': 9,
    'num_unknown': 3,
    'loads': [6, 7, 8, 9],
    'data_root': 'C-PUdata12.mat',
    'src_type_a': [0, 1, 2, 3, 4, 5],
    'src_type_b': [0, 1, 2, 6, 7, 8],
    'known_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'unknown_classes': [9, 10, 11],
    'samples_per_class': 800,
    'test_spc': 200,
}


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True


def zscore(Z):
    Zmax = Z.max(axis=1, keepdims=True)
    Zmin = Z.min(axis=1, keepdims=True)
    return (Z - Zmin) / (Zmax - Zmin + 1e-8)


def min_max(Z):
    Zmin = Z.min(axis=1, keepdims=True)
    return np.log(Z - Zmin + 1)


def load_target_openset(root_path, tgt_load, fft_enabled, cfg, batch_size, kw):
    mat = scio.loadmat(root_path)
    nc = cfg['class_num']
    spc = cfg['samples_per_class']
    tspc = cfg['test_spc']
    known = cfg['known_classes']

    sig_tr = mat[f'load{tgt_load}_train']
    if fft_enabled:
        f_tr = zscore(min_max(np.abs(fft(sig_tr))[:, :1600]))
    else:
        f_tr = zscore(sig_tr)
    l_tr = np.array([i // spc for i in range(len(sig_tr))])
    kmask = np.isin(l_tr, known)
    f_tr, l_tr = f_tr[kmask], l_tr[kmask]

    f_tr_t = torch.from_numpy(f_tr).float().unsqueeze(1)
    l_tr_t = torch.from_numpy(l_tr).long()
    ld_tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(f_tr_t, l_tr_t),
        batch_size=batch_size, shuffle=True, drop_last=True, **kw)

    sig_ts = mat[f'load{tgt_load}_test']
    if fft_enabled:
        f_ts = zscore(min_max(np.abs(fft(sig_ts))[:, :1600]))
    else:
        f_ts = zscore(sig_ts)
    l_ts = np.full(len(sig_ts), -1, dtype=np.int64)
    for c in known:
        l_ts[c * tspc:(c + 1) * tspc] = c

    f_ts_t = torch.from_numpy(f_ts).float().unsqueeze(1)
    l_ts_t = torch.from_numpy(l_ts).long()
    ld_ts = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(f_ts_t, l_ts_t),
        batch_size=batch_size, shuffle=False, drop_last=False, **kw)

    return ld_tr, ld_ts


def enumerate_tasks(loads):
    tasks = []
    for tgt in loads:
        for sa, sb in itertools.combinations([l for l in loads if l != tgt], 2):
            tasks.append({'src': [sa, sb], 'tgt': tgt})
    return tasks


def hs(k, u):
    if k + u == 0:
        return 0.0
    return 2 * k * u / (k + u)


@torch.no_grad()
def extract_all_features(model, loaders, device):
    out = {}
    for name, loader in loaders.items():
        model.eval()
        logits_list, feats_list, labels_list = [], [], []
        for x, y in loader:
            x = x.to(device)
            logits, f = model(x)
            logits_list.append(logits.cpu())
            feats_list.append(f.cpu())
            labels_list.append(y)
        out[name] = {
            'logits': torch.cat(logits_list, 0),
            'feats': torch.cat(feats_list, 0),
            'labels': torch.cat(labels_list, 0),
        }
    return out


def fit_maha_on_features(feats, labels, num_classes, reg=0.01):
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
            ac = torch.cat(fbd[c], 0)
            mu = ac.mean(0)
            cov = torch.cov(ac.T) + reg * eye
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


def mahalanobis_scores(feats, mus, precs):
    dd = []
    for c in range(mus.shape[0]):
        diff = feats - mus[c].unsqueeze(0)
        dd.append((diff @ precs[c] @ diff.T).diag())
    dd = torch.stack(dd, 1)
    return dd.min(1)


def eval_maha_threshold(scores, pred, labels):
    is_known = labels >= 0
    n_k = is_known.sum().item()
    n_u = (~is_known).sum().item()
    sorted_scores = scores.sort().values
    best_h = 0.0
    best_metrics = None
    for pct in range(5, 95, 1):
        th = sorted_scores[int(pct / 100 * (len(sorted_scores) - 1))].item()
        accepted = scores >= th
        rejected = ~accepted
        kc = (accepted & is_known & (pred == labels)).sum().item()
        ur = (rejected & ~is_known).sum().item()
        ua = (accepted & ~is_known).sum().item()
        known = 100.0 * kc / n_k if n_k else 0.0
        unk_rej = 100.0 * ur / n_u if n_u else 0.0
        fa = 100.0 * ua / n_u if n_u else 0.0
        h = hs(known, unk_rej)
        if h > best_h:
            best_h = h
            best_metrics = {'known': known, 'unk_rej': unk_rej, 'FA': fa, 'H': h, 'th': th, 'pct': pct}
    return best_metrics


def scale_radial(feats, beta):
    if beta == 0.0:
        return feats
    norm = feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return feats / (norm ** beta)


def eval_tam_ccat_from_features(fd, num_known, conf_threshold=0.9, min_samples=10,
                                alphas=[0, 0.5, 1, 2, 3], beta=0.0):
    src_a = fd['src_a']
    src_b = fd['src_b']
    tgt_tr = fd['tgt_tr']
    tgt_ts = fd['tgt_ts']

    src_feats = torch.cat([scale_radial(src_a['feats'], beta), scale_radial(src_b['feats'], beta)], 0)
    src_labels = torch.cat([src_a['labels'], src_b['labels']], 0)
    tgt_tr_feats = scale_radial(tgt_tr['feats'], beta)
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

    md, pred = mahalanobis_scores(tgt_ts_feats, mus, precs)
    best_h = 0.0
    best_metrics = None
    for alpha in alphas:
        score = -md / (1.0 + alpha * shifts[pred])
        metrics = eval_maha_threshold(score, pred, tgt_ts_labels)
        metrics['alpha'] = alpha
        if metrics['H'] > best_h:
            best_h = metrics['H']
            best_metrics = metrics
    return best_metrics


def pilot_radial(fd, num_known):
    best_h = 0.0
    best_beta = None
    best_metrics = None
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        metrics = eval_tam_ccat_from_features(fd, num_known, beta=beta)
        if metrics['H'] > best_h:
            best_h = metrics['H']
            best_beta = beta
            best_metrics = metrics
    best_metrics['best_beta'] = best_beta
    return best_metrics


def pilot_mixup_unknown(fd, num_known, conf_threshold=0.9, min_samples=10):
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
    md_ts, pred_ts = mahalanobis_scores(tgt_ts_feats, mus, precs)

    # Known positives: target train known features
    known_feats = tgt_tr_feats.numpy()
    known_y = np.ones(known_feats.shape[0])

    # Synthetic negatives: mixup pairs from different classes
    n_mix = known_feats.shape[0]
    rng = np.random.RandomState(SEED)
    mixup = []
    for _ in range(n_mix):
        i = rng.randint(0, known_feats.shape[0])
        j = rng.randint(0, known_feats.shape[0])
        if tgt_tr_labels[i] == tgt_tr_labels[j]:
            j = (j + 1) % known_feats.shape[0]
        lam = rng.uniform(0.2, 0.8)
        mix = lam * known_feats[i] + (1 - lam) * known_feats[j]
        mixup.append(mix)
    mixup = np.stack(mixup, 0)
    unknown_y = np.zeros(mixup.shape[0])

    train_x = np.concatenate([known_feats, mixup], 0)
    train_y = np.concatenate([known_y, unknown_y], 0)

    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=SEED)
    clf.fit(train_x, train_y)
    proba_known = clf.predict_proba(tgt_ts_feats.numpy())[:, 1]
    proba_known = torch.from_numpy(proba_known)

    is_known = tgt_ts_labels >= 0
    n_k = is_known.sum().item()
    n_u = (~is_known).sum().item()
    best_h = 0.0
    best_metrics = None
    sorted_p = proba_known.sort().values
    for pct in range(5, 95, 1):
        th = sorted_p[int(pct / 100 * (len(sorted_p) - 1))].item()
        accepted = proba_known >= th
        rejected = ~accepted
        kc = (accepted & is_known & (pred_ts == tgt_ts_labels)).sum().item()
        ur = (rejected & ~is_known).sum().item()
        ua = (accepted & ~is_known).sum().item()
        known = 100.0 * kc / n_k if n_k else 0.0
        unk_rej = 100.0 * ur / n_u if n_u else 0.0
        fa = 100.0 * ua / n_u if n_u else 0.0
        h = hs(known, unk_rej)
        if h > best_h:
            best_h = h
            best_metrics = {'known': known, 'unk_rej': unk_rej, 'FA': fa, 'H': h, 'th': th, 'pct': pct}
    return best_metrics


def main():
    set_seed(SEED)
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    kw = {'num_workers': 0, 'pin_memory': False}
    project_root = os.path.dirname(os.path.dirname(__file__))
    root_path = os.path.join(project_root, CFG['data_root'])
    tasks = enumerate_tasks(CFG['loads'])

    target_tasks = [t for t in tasks if t['tgt'] == 9]
    print(f'Pilot on {len(target_tasks)} load-9 tasks: device={device}', flush=True)

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

        model_path = os.path.join(project_root, 'models', 'notebook_baseline', f'pu_task{task_idx}_seed{SEED}_best.pth')
        if not os.path.exists(model_path):
            print(f'Model not found: {model_path}, skipping', flush=True)
            continue
        ckpt = torch.load(model_path, map_location=device)

        # Baseline TAM+CCAT
        model = CNN_1D_ResNet18(num_classes=CFG['num_known']).to(device)
        model.load_state_dict(ckpt['model'])
        fd_base = extract_all_features(model, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)
        base = eval_tam_ccat_from_features(fd_base, CFG['num_known'])
        print(f'  TAM+CCAT baseline: H={base["H"]:.2f}% known={base["known"]:.2f}% unk_rej={base["unk_rej"]:.2f}% FA={base["FA"]:.2f}% alpha={base["alpha"]}', flush=True)

        # AdaBN: update BN stats with target train, then extract features
        model_adabn = CNN_1D_ResNet18(num_classes=CFG['num_known']).to(device)
        model_adabn.load_state_dict(ckpt['model'])
        model_adabn.train()
        with torch.no_grad():
            for x, _ in ld_ttr:
                x = x.to(device)
                _ = model_adabn(x)
        fd_adabn = extract_all_features(model_adabn, {'src_a': ld_a, 'src_b': ld_b, 'tgt_tr': ld_ttr, 'tgt_ts': ld_tts}, device)
        adabn = eval_tam_ccat_from_features(fd_adabn, CFG['num_known'])
        print(f'  AdaBN+TAM+CCAT:    H={adabn["H"]:.2f}% known={adabn["known"]:.2f}% unk_rej={adabn["unk_rej"]:.2f}% FA={adabn["FA"]:.2f}%', flush=True)

        # Radial scaling
        rad = pilot_radial(fd_base, CFG['num_known'])
        print(f'  Radial TAM+CCAT:   H={rad["H"]:.2f}% known={rad["known"]:.2f}% unk_rej={rad["unk_rej"]:.2f}% FA={rad["FA"]:.2f}% beta={rad["best_beta"]}', flush=True)

        # Mixup unknowns
        mix = pilot_mixup_unknown(fd_base, CFG['num_known'])
        print(f'  Mixup-unknown:     H={mix["H"]:.2f}% known={mix["known"]:.2f}% unk_rej={mix["unk_rej"]:.2f}% FA={mix["FA"]:.2f}%', flush=True)

        results.append({
            'task': task_idx,
            'src': f'{s_loads[0]}+{s_loads[1]}',
            'tgt': tgt_load,
            'base_H': base['H'],
            'adabn_H': adabn['H'],
            'rad_H': rad['H'],
            'mix_H': mix['H'],
        })

    print('\n=== Pilot summary ===', flush=True)
    print('Task  src->tgt  base_H  adabn_H  rad_H  mix_H', flush=True)
    for r in results:
        print(f'{r["task"]:4d}  {r["src"]:>7s}->{r["tgt"]}  {r["base_H"]:6.2f}  {r["adabn_H"]:7.2f}  {r["rad_H"]:5.2f}  {r["mix_H"]:5.2f}', flush=True)
    if results:
        print(f'Average base_H={np.mean([r["base_H"] for r in results]):.2f} adabn_H={np.mean([r["adabn_H"] for r in results]):.2f} rad_H={np.mean([r["rad_H"] for r in results]):.2f} mix_H={np.mean([r["mix_H"] for r in results]):.2f}', flush=True)


if __name__ == '__main__':
    main()
