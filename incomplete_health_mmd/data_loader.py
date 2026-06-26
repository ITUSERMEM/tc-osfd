"""
Data loaders for incomplete health state domain adaptation.

Each source domain is loaded with a *subset* of classes (keep_classes).
Labels are kept in the *original* class space so the classifier's output
dimension matches the total number of classes in the dataset.
"""

import numpy as np
import torch
import scipy.io as scio
from scipy.fftpack import fft
from scipy import signal


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def order_tracking_fft(signals, rpm, fs=64000, n_angles=3200, n_orders=1600):
    """
    Time-domain angle resampling → order spectrum.

    Resamples raw vibration signal from constant-Δt to constant-Δθ
    using nominal RPM, then computes FFT. The output spectrum has
    peaks at fixed *orders* (multiples of shaft speed), independent of RPM.

    signals:   (N, L) raw vibration, shape (n_samples, signal_length)
    rpm:       nominal rotational speed (e.g. 1500, 900)
    fs:        sampling rate in Hz (default 64000 for PU)
    n_angles:  number of equi-angular resampling points
    n_orders:  output order spectrum length (truncated FFT)

    Returns: (N, n_orders) order spectrum magnitude
    """
    n_samples, L = signals.shape
    # Time axis
    t = np.arange(L) / fs
    # Angular axis: theta = 2π * (rpm/60) * t
    theta = 2.0 * np.pi * (rpm / 60.0) * t
    # Equi-angular grid
    theta_new = np.linspace(theta[0], theta[-1], n_angles)

    order_specs = np.zeros((n_samples, n_orders), dtype=np.float64)
    for i in range(n_samples):
        sig = signals[i].astype(np.float64)
        # Angle-domain resampling
        sig_angle = np.interp(theta_new, theta, sig)
        # FFT → order spectrum
        spec = np.abs(fft(sig_angle))[:n_orders]
        order_specs[i] = spec
    return order_specs


def speed_normalize_fft(fft_data, src_rpm=1500, tgt_rpm=900):
    """
    Stretch FFT frequency axis to normalize speed differences.

    Maps a tgt_rpm FFT spectrum to src_rpm-equivalent order space
    by linearly stretching the frequency axis by ratio = src_rpm / tgt_rpm.

    fft_data: (N, D) numpy array of FFT magnitude spectra
    Returns: (N, D) speed-normalized spectra
    """
    ratio = src_rpm / tgt_rpm
    D = fft_data.shape[1]
    src_positions = np.arange(D, dtype=np.float64)
    query_positions = src_positions / ratio
    normalized = np.array([
        np.interp(src_positions, query_positions, row, left=0.0, right=0.0)
        for row in fft_data
    ])
    return normalized


def zscore(Z):
    Zmax = Z.max(axis=1, keepdims=True)
    Zmin = Z.min(axis=1, keepdims=True)
    Z = (Z - Zmin) / (Zmax - Zmin)
    return Z


def min_max(Z):
    Zmin = Z.min(axis=1, keepdims=True)
    Z = np.log(Z - Zmin + 1)
    return Z


# ---------------------------------------------------------------------------
# Domain loader with class mask (original labels preserved)
# ---------------------------------------------------------------------------

# PU 12-class → 4 damage-type mapping
MAP_12_TO_4 = [0, 0, 1, 2, 2, 3, 3, 1, 2, 2, 3, 3]


class FSDRDataset(torch.utils.data.Dataset):
    """Frequency-domain amplitude randomization (FSDR) applied on the fly.

    Stores raw FFT magnitude spectra and applies per-sample multiplicative
    amplitude noise before normalization. This simulates load-dependent
    amplitude variation without altering the phase/frequency structure.
    """

    def __init__(self, raw_specs, labels, fsdr_alpha, fsdr_prob=1.0):
        self.raw_specs = raw_specs.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.fsdr_alpha = fsdr_alpha
        self.fsdr_prob = fsdr_prob

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.raw_specs[idx].copy()
        if self.fsdr_alpha > 0 and np.random.rand() < self.fsdr_prob:
            noise = np.random.randn(spec.shape[0]).astype(np.float32)
            spec = spec * (1.0 + self.fsdr_alpha * noise)
            spec = np.maximum(spec, 0.0)
        spec = spec.reshape(1, -1)
        spec = zscore(min_max(spec)).squeeze(0)
        return torch.from_numpy(spec).float().unsqueeze(0), torch.tensor(
            self.labels[idx], dtype=torch.long)


def speed_augment(signals, scale_range=(0.55, 1.0), seed=None):
    """
    Time-stretch each 1-D signal to simulate a different shaft speed.

    scale_range defines the target_speed / original_speed ratio.
    For original 25 Hz, scale=0.6 produces a 15-Hz-like sample.
    The output has the same length L as the input.  For each sample a random
    crop is taken from the time-stretched signal so that different augmented
    copies use different portions of the original signal.
    Returns an augmented copy; original signals are untouched.
    """
    rng = np.random.default_rng(seed)
    N, L = signals.shape
    out = np.empty_like(signals, dtype=np.float64)
    idx = np.arange(L, dtype=np.float64)
    for i in range(N):
        s = rng.uniform(scale_range[0], scale_range[1])
        max_offset = (L - 1) * (1.0 - s)
        offset = rng.uniform(0.0, max(max_offset, 0.0))
        query = idx * s + offset
        if scale_range[1] > 1.0:
            query = query % (L - 1)
        out[i] = np.interp(query, idx, signals[i].astype(np.float64))
    return out


def freq_scale_augment(raw_specs, scale_range=(0.55, 1.0), seed=None):
    """
    Scale the frequency axis of FFT magnitude spectra to simulate a different
    shaft speed.  For original speed f0 and target speed s*f0, every fault
    harmonic at bin k moves to bin s*k.

    raw_specs: (N, L) numpy array of FFT magnitudes.
    Returns an augmented copy of the same shape.
    """
    rng = np.random.default_rng(seed)
    N, L = raw_specs.shape
    out = np.empty_like(raw_specs, dtype=np.float64)
    idx = np.arange(L, dtype=np.float64)
    for i in range(N):
        s = rng.uniform(scale_range[0], scale_range[1])
        # Sample old spectrum at positions idx / s; out-of-range → 0.
        query = idx / s
        out[i] = np.interp(query, idx, raw_specs[i].astype(np.float64),
                           left=0.0, right=0.0)
    return out


def load_source_domain(root_path, var_name, fft_enabled, class_num,
                        samples_per_class, keep_classes, batch_size, kwargs,
                        remap_fn=None, augment_speed=False,
                        speed_scale_range=(0.55, 1.0), speed_seed=None,
                        speed_replace_prob=0.5, augment_mode='freq',
                        fsdr_alpha=0.0, fsdr_prob=1.0):
    """
    Load one source domain, keeping only *keep_classes*.

    If remap_fn is provided (e.g. MAP_12_TO_4), labels are first computed
    in the original label space, then remapped. *keep_classes* must be
    expressed in the remapped label space.

    Args:
        augment_speed: if True, randomly replace source samples with a
                       speed-augmented version (simulates lower/higher shaft
                       speeds). Probability controlled by speed_replace_prob.
        speed_scale_range: (min, max) target_speed / original_speed ratio.
        speed_seed: seed for the per-sample random speed factors.
        speed_replace_prob: probability of replacing a sample with its
                            speed-augmented copy.
        fsdr_alpha: if >0, apply frequency-domain amplitude randomization
                    with this noise magnitude on the FFT spectra.
        fsdr_prob: probability of applying FSDR to each sample.

    Returns:
        DataLoader yielding (features, labels)
            features: (B, 1, L)  float32
            labels:   (B,)       int64
    """
    mat_data = scio.loadmat(root_path)
    signals = mat_data[var_name]

    # Apply time-domain speed augmentation BEFORE FFT if requested.
    if augment_speed and augment_mode == 'time':
        aug_signals1 = speed_augment(signals, speed_scale_range, speed_seed)
        aug_signals2 = speed_augment(signals, speed_scale_range, speed_seed + 1)
        signals = np.concatenate([signals, aug_signals1, aug_signals2], axis=0)

    if fft_enabled:
        raw_specs = np.abs(fft(signals))[:, :1600]
        if augment_speed and augment_mode in ('freq', 'freq_low'):
            rng = np.random.default_rng(speed_seed)
            if augment_mode == 'freq_low':
                low_bins = 400
                low_part = raw_specs[:, :low_bins]
                high_part = raw_specs[:, low_bins:]
                aug_low = freq_scale_augment(low_part, speed_scale_range, speed_seed)
                mask = rng.random(raw_specs.shape[0]) < speed_replace_prob
                low_part = np.where(mask[:, None], aug_low, low_part)
                raw_specs = np.concatenate([low_part, high_part], axis=1)
            else:
                aug_specs = freq_scale_augment(raw_specs, speed_scale_range, speed_seed)
                mask = rng.random(raw_specs.shape[0]) < speed_replace_prob
                raw_specs = np.where(mask[:, None], aug_specs, raw_specs)
        features = raw_specs
    else:
        features = signals

    n = features.shape[0]

    # Build raw labels in original class space.
    # Data is stored as contiguous class blocks of samples_per_class rows,
    # repeated class_num times. When augmentation concatenates copies, we
    # wrap the index back into one original cycle.
    cycle_len = samples_per_class * class_num
    raw_labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        raw_labels[i] = (i % cycle_len) // samples_per_class

    # Apply remapping (e.g. 12→4 for pu4)
    if remap_fn is not None:
        labels_4 = np.array([remap_fn[l] for l in raw_labels], dtype=np.int64)
    else:
        labels_4 = raw_labels

    # Filter by keep_classes (in remapped space)
    mask = np.isin(labels_4, keep_classes)

    features = features[mask]
    labels = labels_4[mask]

    if fft_enabled and fsdr_alpha > 0:
        dataset = FSDRDataset(features, labels, fsdr_alpha, fsdr_prob)
    else:
        if fft_enabled:
            features = zscore(min_max(features))
        else:
            features = zscore(features)
        features = torch.from_numpy(features).float().unsqueeze(1)
        labels = torch.from_numpy(labels).long()
        dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, drop_last=True,
                                          **kwargs)
    return loader


def load_target_train(root_path, var_name, fft_enabled, class_num,
                       samples_per_class, batch_size, kwargs,
                       remap_fn=None):
    """
    Load target *train* set (all classes). Used for MMD alignment.
    If remap_fn is provided, labels are remapped (e.g. 12→4 for pu4).
    """
    mat_data = scio.loadmat(root_path)
    signals = mat_data[var_name]

    if fft_enabled:
        features = zscore(min_max(np.abs(fft(signals))))[:, :1600]
    else:
        features = zscore(signals)

    n = features.shape[0]
    orig_samples_per_class = 800
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        labels[i] = i // orig_samples_per_class

    if remap_fn is not None:
        labels = np.array([remap_fn[l] for l in labels], dtype=np.int64)

    features = torch.from_numpy(features).float().unsqueeze(1)
    labels = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, drop_last=True,
                                          **kwargs)
    return loader


def load_target_test(root_path, var_name, fft_enabled, class_num,
                      test_samples_per_class, batch_size, kwargs,
                      remap_fn=None):
    """
    Load target *test* set (all classes). Used for evaluation.
    If remap_fn is provided, labels are remapped (e.g. 12→4 for pu4).
    """
    mat_data = scio.loadmat(root_path)
    signals = mat_data[var_name]

    if fft_enabled:
        features = zscore(min_max(np.abs(fft(signals))))[:, :1600]
    else:
        features = zscore(signals)

    n = features.shape[0]
    orig_samples_per_class = 800
    test_orig = 200
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        labels[i] = i // test_orig

    if remap_fn is not None:
        labels = np.array([remap_fn[l] for l in labels], dtype=np.int64)

    features = torch.from_numpy(features).float().unsqueeze(1)
    labels = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, drop_last=False,
                                          **kwargs)
    return loader
