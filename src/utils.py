#author:zhaochao time:2021/5/18

import torch as t
import torch.nn.functional as F
import numpy as np
import  random
import torch.nn as nn

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = t.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = t.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [t.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def guassian_kernel_aw(source, target, kernel_mul=2.0, kernel_num=5):
    """
    Activation-Weighted RBF kernel.

    Distance between sample i and j is weighted by the product of their
    per-dimension activations:  w_s[d] * w_t[d] where w = |x[d]| / ||x||.

    Dimensions where EITHER sample is near-zero activate are automatically
    silenced — handles support-set shift across domains.
    """
    n_s = source.size(0)
    n_t = target.size(0)
    n_all = n_s + n_t
    total = t.cat([source, target], dim=0)   # [N, D]

    # Per-sample activation weights: w[i,d] = |x[i,d]| / ||x[i]||
    w = t.abs(total) / (total.norm(dim=1, keepdim=True) + 1e-8)  # [N, D]

    # Pairwise L2 distance (unweighted, for bandwidth estimation)
    total0 = total.unsqueeze(0).expand(n_all, n_all, total.size(1))
    total1 = total.unsqueeze(1).expand(n_all, n_all, total.size(1))
    diff = total0 - total1                                     # [N, N, D]
    L2_unweighted = (diff ** 2).sum(2)                         # [N, N]

    # Activation-weighted L2:
    #   L2_AW[i,j] = Σ_d w[i,d] · w[j,d] · (x[i,d] - x[j,d])²
    w_i = w.unsqueeze(0)  # [1, N, D]
    w_j = w.unsqueeze(1)  # [N, 1, D]
    L2_aw = ((w_i * w_j) * (diff ** 2)).sum(2)  # [N, N]

    # Bandwidth from unweighted L2 (same as standard)
    bandwidth = t.sum(L2_unweighted.data) / (n_all**2 - n_all)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    K = t.zeros(n_all, n_all, device=source.device)
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        K += t.exp(-L2_aw / bw)
    return K


def mmd_vmf(source, target, kappa=10.0, kernel_num=5):
    """
    von Mises-Fisher kernel MMD on L2-normalized features.

    K(x,y) = exp(κ · x̂ᵀŷ)  where x̂ = x/||x||.

    Operates purely in angular space — tolerant to angular deviations
    caused by cross-speed harmonic shift. κ controls concentration:
      κ large → tight angular alignment (for same-speed)
      κ small → tolerant angular alignment (for cross-speed)

    Uses kernel_num concentration levels (κ, κ/2, κ/4, ...) like standard
    MMD uses multiple bandwidths.
    """
    # L2 normalize
    source = source / (source.norm(dim=1, keepdim=True) + 1e-8)
    target = target / (target.norm(dim=1, keepdim=True) + 1e-8)

    n_s = source.size(0)
    n_t = target.size(0)
    n_all = n_s + n_t
    total = t.cat([source, target], dim=0)  # [N, D]

    # Cosine similarity matrix
    cos_sim = total @ total.T  # [N, N]

    # Multi-κ kernel: Σ_i exp(κ_i · cos_sim)
    K = t.zeros(n_all, n_all, device=source.device)
    for i in range(kernel_num):
        ki = kappa / (2 ** i)
        K += t.exp(ki * cos_sim)

    XX = K[:n_s, :n_s]
    YY = K[n_s:, n_s:]
    XY = K[:n_s, n_s:]
    YX = K[n_s:, :n_s]
    return (XX.mean() + YY.mean() - XY.mean() - YX.mean())


def mmd_cycle_l2(source, target, max_shift=3, tau=0.01, kernel_mul=2.0, kernel_num=5):
    """
    L2-normalized MMD with cycle-shift tolerance.

    Before computing the kernel, L2-normalize features, then try cyclically
    shifting target features by {-max_shift..+max_shift} positions.
    Uses soft-min over shifts — backbone learns to produce features that
    align under the best shift, compensating for speed-induced harmonic shift.
    """
    # L2 normalize
    source = source / (source.norm(dim=1, keepdim=True) + 1e-8)
    target = target / (target.norm(dim=1, keepdim=True) + 1e-8)

    n_s = source.size(0)
    n_t = target.size(0)
    n_all = n_s + n_t
    total = t.cat([source, target], dim=0)

    # SS + TT: no shift (same-domain)
    total0 = total.unsqueeze(0).expand(n_all, n_all, total.size(1))
    total1 = total.unsqueeze(1).expand(n_all, n_all, total.size(1))
    L2_full = ((total0 - total1) ** 2).sum(2)  # [N, N]

    # ST block: try all shifts, soft-min
    L2_st_list = []
    for s in range(-max_shift, max_shift + 1):
        tgt_s = t.roll(target, shifts=s, dims=1)
        src0 = source.unsqueeze(1).expand(n_s, n_t, source.size(1))
        tgt1 = tgt_s.unsqueeze(0).expand(n_s, n_t, target.size(1))
        l2_s = ((src0 - tgt1) ** 2).sum(2)
        L2_st_list.append(l2_s)

    L2_st_stack = t.stack(L2_st_list, dim=0)  # [S, n_s, n_t]
    # Soft-min: L2_st_min ≈ min_s L2_st_stack[s]
    L2_st_min = -tau * t.logsumexp(-L2_st_stack / tau, dim=0)

    L2 = L2_full.clone()
    L2[:n_s, n_s:] = L2_st_min
    L2[n_s:, :n_s] = L2_st_min.T

    # Bandwidth
    bandwidth = t.sum(L2.data) / (n_all**2 - n_all)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    K = t.zeros(n_all, n_all, device=source.device)
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        K += t.exp(-L2 / bw)

    XX = K[:n_s, :n_s]
    YY = K[n_s:, n_s:]
    XY = K[:n_s, n_s:]
    YX = K[n_s:, :n_s]
    return (XX.mean() + YY.mean() - XY.mean() - YX.mean())


def mmd_aw(source, target, kernel_mul=2.0, kernel_num=5):
    """MMD with Activation-Weighted RBF kernel."""
    n_s = source.size(0)
    n_t = target.size(0)
    kernels = guassian_kernel_aw(source, target, kernel_mul=kernel_mul,
                                  kernel_num=kernel_num)
    XX = kernels[:n_s, :n_s]
    YY = kernels[n_s:, n_s:]
    XY = kernels[:n_s, n_s:]
    YX = kernels[n_s:, :n_s]
    return (XX.mean() + YY.mean() - XY.mean() - YX.mean())


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5,
                           fix_sigma=None, normalize=False, env_pool=None):
    """
    MMD with RBF kernel. Optional modifications (internal to kernel):
      normalize: L2-normalize features before kernel computation
      env_pool:  avg_pool1d kernel_size before kernel (spectral envelope)
    """
    if env_pool is not None:
        source = t.nn.functional.avg_pool1d(
            source.unsqueeze(1), kernel_size=env_pool, stride=env_pool
        ).squeeze(1)
        target = t.nn.functional.avg_pool1d(
            target.unsqueeze(1), kernel_size=env_pool, stride=env_pool
        ).squeeze(1)
    if normalize:
        source = source / (source.norm(dim=1, keepdim=True) + 1e-8)
        target = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    n_source = int(source.size()[0])
    n_target = int(target.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n_source, :n_source]
    YY = kernels[n_source:, n_source:]
    XY = kernels[:n_source, n_source:]
    YX = kernels[n_source:, :n_source]
    loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()

    return loss


def mmd_class_conditional(feat_src, feat_tgt, labels_src, labels_tgt, num_known,
                          min_samples=5):
    """Per-class MMD: align source/target distributions within each known class.

    For every class k, extract source features with label k and target features
    with label k, compute RBF-MMD between them, and average over classes that
    have at least min_samples samples in both domains. Classes missing from a
    batch are skipped. The zero fallback is written to preserve the computation
    graph so gradient still flows through the feature extractor.
    """
    losses = []
    for k in range(num_known):
        mask_s = labels_src == k
        mask_t = labels_tgt == k
        if mask_s.sum() >= min_samples and mask_t.sum() >= min_samples:
            losses.append(mmd_rbf_noaccelerate(
                feat_src[mask_s], feat_tgt[mask_t],
                kernel_mul=2.0, kernel_num=5))
    if not losses:
        return feat_src.sum() * 0.0
    return t.stack(losses).mean()


def mmd_env(source, target, pool_size=4, normalize=True,
             kernel_mul=2.0, kernel_num=5):
    """Env-MMD: spectral-envelope MMD (pool + L2-norm + RBF)."""
    return mmd_rbf_noaccelerate(source, target, kernel_mul=kernel_mul,
                                  kernel_num=kernel_num, normalize=normalize,
                                  env_pool=pool_size)


def mmd_l2norm(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """MMD on L2-normalized features — aligns angular distributions."""
    return mmd_rbf_noaccelerate(source, target, kernel_mul=kernel_mul,
                                  kernel_num=kernel_num, fix_sigma=fix_sigma,
                                  normalize=True)


def cmmd_loss(feat_src, feat_tgt, labels_src, kernel_mul=2.0, kernel_num=5):
    """
    Conditioned MMD: type-conditioned bandwidth via scaling factors.

    Uses ONE global sigma_med (like standard MMD), then applies per-type
    temperature scaling inside the exponential:
        K_ij = exp(-d_ij² / (sigma * tau_type))
      tau_ss_same = 0.5   tighter kernel for same-class → compactness
      tau_ss_diff = 2.0   looser kernel for diff-class → separation
      tau_st      = 1.0   default for cross-domain

    Returns scalar — interface identical to mmd_rbf_noaccelerate.
    """
    n_s = feat_src.size(0)
    n_t = feat_tgt.size(0)
    n_all = n_s + n_t
    total = t.cat([feat_src, feat_tgt], dim=0)

    # Global sigma_med (same as standard MMD)
    total0 = total.unsqueeze(0).expand(n_all, n_all, total.size(1))
    total1 = total.unsqueeze(1).expand(n_all, n_all, total.size(1))
    L2 = ((total0 - total1) ** 2).sum(2)  # [n_all, n_all]
    sigma_med = L2.mean()
    sigma_base = sigma_med / (kernel_mul ** (kernel_num // 2))

    # --- vectorized masks ---
    # SS same-class: [n_s, n_s]
    y_s = labels_src.view(1, n_s)
    mask_ss_same = (y_s == y_s.T)  # [n_s, n_s]

    # SS diff-class: [n_s, n_s]
    mask_ss_diff = (y_s != y_s.T)

    # ST: use broadcasting (no explicit mask needed, handled via index slicing)

    # --- tau factors ---
    tau_ss_same = 0.5
    tau_ss_diff = 2.0
    tau_st      = 1.0

    # --- build kernel ---
    K = t.zeros(n_all, n_all, device=feat_src.device)
    for i in range(kernel_num):
        bw = sigma_base * (kernel_mul ** i)

        # SS same-class
        bw_same = bw * tau_ss_same
        K[:n_s, :n_s] += mask_ss_same.float() * t.exp(-L2[:n_s, :n_s] / bw_same)

        # SS diff-class
        bw_diff = bw * tau_ss_diff
        K[:n_s, :n_s] += mask_ss_diff.float() * t.exp(-L2[:n_s, :n_s] / bw_diff)

        # ST cross-domain
        bw_st = bw * tau_st
        K[:n_s, n_s:] += t.exp(-L2[:n_s, n_s:] / bw_st)
        K[n_s:, :n_s] += t.exp(-L2[n_s:, :n_s] / bw_st)

    # Standard MMD formula
    XX = K[:n_s, :n_s]
    YY = K[n_s:, n_s:]
    XY = K[:n_s, n_s:]
    YX = K[n_s:, :n_s]
    return (XX.mean() + YY.mean() - XY.mean() - YX.mean())


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = t.mean(source, 1, keepdim=True) - source
    xc = t.matmul(t.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = t.mean(target, 1, keepdim=True) - target
    xct = t.matmul(t.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = t.mean(t.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*4)
    return loss




class Center_loss(nn.Module):
    def __init__(self,src_class):
        super(Center_loss, self).__init__()

        self.n_class=src_class
        self.MSELoss = nn.MSELoss()



    def forward(self, s_feature,s_labels):


        n, d = s_feature.shape

        # get labels


        # image number in each class
        ones = t.ones_like(s_labels, dtype=t.float)
        zeros = t.zeros(self.n_class, device=s_labels.device)

        s_n_classes = zeros.scatter_add(0, s_labels, ones)


        # image number cannot be 0, when calculating centroids
        ones = t.ones_like(s_n_classes)
        s_n_classes = t.max(s_n_classes, ones)


        # calculating centroids, sum and divide
        zeros = t.zeros(self.n_class, d, device=s_feature.device)

        s_sum_feature = zeros.scatter_add(0, t.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)

        s_centroid = t.div(s_sum_feature, s_n_classes.view(self.n_class, 1))


        # calculating inter distance

        temp = t.zeros((n, d), device=s_feature.device)

        for i in range(n):
            temp[i] = s_centroid[s_labels[i]]

       #
        # intra_loss = t.norm(temp-s_feature, p=1, dim=0).sum()
        # intra_loss = intra_loss / (d * n)

        #### way 1:
        intra_loss = self.MSELoss(temp, s_feature)



        return intra_loss




class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = t.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = t.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = t.norm(anchor - neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = t.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = t.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()# 计算距离
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)#返回对角线索引
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg



def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)  # cpu
    t.cuda.manual_seed_all(seed)  # 并行gpu
    t.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    t.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速




def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1) / 2
    else:
        sim = F.pairwise_distance(x1, x2) / t.norm(x2, dim=1)
    return sim




def crit_contrast(feats, probs, s_ctds, t_ctds, lambd=1e-3):
    batch_num = feats.shape[0]
    class_num = s_ctds.shape[0]
    probs = F.softmax(probs, dim=-1)
    max_probs, preds = probs.max(1, keepdim=True)
    # print(probs.shape, max_probs.shape)
    select_index = t.nonzero(max_probs.squeeze() >= 0.3).squeeze(1)
    select_index = select_index.cpu().tolist()

    # todo: calculate margins
    # dist_ctds = cal_cossim(to_np(s_ctds), to_np(t_ctds))
    dist_ctds = cal_sim(s_ctds, t_ctds)
    # print('dist_ctds', dist_ctds.shape)

    M = np.ones(class_num)
    for i in range(class_num):
        # M[i] = np.sum(dist_ctds[i, :]) - dist_ctds[i, i]
        M[i] = dist_ctds.mean() - dist_ctds[i]
        M[i] /= class_num - 1
    # print('M', M)

    # todo: calculate D_k between known samples to its source centroid &
    # todo: calculate D_u distances between unknown samples to all source centroids
    D_k, n_k = 0, 1e-5
    D_u, n_u = 0, 1e-5
    for i in select_index:
        class_id = preds[i][0]
        if class_id < class_num:
            # D_k += F.pairwise_distance(feats[i, :], s_ctds[class_id]).squeeze()
            # print(feats.shape, i)
            D_k += cal_sim(feats[i, :], s_ctds[class_id, :])
            # print('D_k', D_k)
            n_k += 1
        else:
            # todo: judge if unknown sample in the radius region of known centroid
            rp_feats = feats[i, :].unsqueeze(0).repeat(class_num, 1)

            # dist_known = F.pairwise_distance(rp_feats, s_ctds)
            dist_known = cal_sim(rp_feats, s_ctds)
            # print('dist_known', len(dist_known), dist_known)

            M_mean = M.mean()
            outliers = dist_known < M_mean
            dist_margin = (dist_known - M_mean) * outliers.float()
            D_u += dist_margin.sum()

    loss = D_k / n_k  # - D_u / n_u
    return loss.mean() * lambd



def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * t.log(predict_prob + epsilon)
    return t.sum(instance_level_weight * ce * class_level_weight) / float(N)

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    mask = predict_prob.ge(0.000001)  # 逐元素比较
    mask_out = t.masked_select(predict_prob, mask)#


    entropy =-mask_out * t.log(mask_out)


#
    return t.sum(instance_level_weight * entropy * class_level_weight) / float(N)

#

def normalization(input):
    kethe=0.0000000001
    output=(input-min(input))/(max(input)-min(input)+kethe)

    return output




def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None,
                                  epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'


    bce = -label * t.log(predict_prob + epsilon) - (1.0 - label) * t.log(1.0 - predict_prob + epsilon)

    return t.sum(instance_level_weight * bce * class_level_weight) / float(N)


# ===================================================================
# Gradient Reversal Layer + Speed Discriminator
# ===================================================================

class GradReverse(t.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky 2016)."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SpeedDiscriminator(nn.Module):
    """2-class MLP: predicts source(1500rpm) vs target(900rpm) from features."""
    def __init__(self, feat_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)


def adv_lambda(progress, max_lambda=1.0):
    """Ganin-style annealing: λ = 2/(1+exp(-10·p)) - 1, scaled by progress."""
    p = progress  # ∈ [0, 1]
    return max_lambda * (2.0 / (1.0 + t.exp(t.tensor(-10.0 * p))) - 1.0)


class SupConLoss(nn.Module):
    """Supervised contrastive loss with same-class positives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        device = features.device
        batch_size = features.size(0)

        similarity = t.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = t.eq(labels, labels.T).float().to(device)

        eye_mask = t.eye(batch_size, device=device)
        logits_mask = t.ones_like(mask) - eye_mask
        mask = mask * logits_mask

        exp_sim = t.exp(similarity) * logits_mask
        log_prob = similarity - t.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        loss = -mean_log_prob_pos[pos_count > 0].mean()
        return loss