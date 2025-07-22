import os
import torch
import numpy as np

CPUSVD = True
EPS = 1e-8

_global_svd_fail_counter = 0

# Loss1 : net pred 9D, then use this function to convert to rot matrix, finally compute L1 loss with GT matrix
def transform_rot9D_to_matrix(A):
    pred = A.reshape(-1, 3, 3)
    device = A.device
    if CPUSVD:
        pred = pred.cpu()
    try:
        U, S, VT = torch.linalg.svd(pred)
        with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            s3sign = torch.det(torch.matmul(U, VT))
        diag = torch.stack((torch.ones_like(s3sign), torch.ones_like(s3sign), s3sign), -1)
        diag = torch.diag_embed(diag)

        pred_orth = U @ diag @ VT
        pred_orth = pred_orth.to(device)
    except:
        pred_orth = torch.eye(3, dtype=torch.float, device=device)[None].repeat_interleave(pred.shape[0], 0)
    return pred_orth

#Loss2: Fisher Loss
def fisher_loss(net_out, R, reduce=True, overreg=1.025):
    A = net_out.view(-1, 3, 3)
    loss_v = KL_Fisher(A, R, overreg=overreg)
    if loss_v is None:
        loss_v = 0 * A[:, 0, 0]
    if reduce:
        loss_v = loss_v.mean()
    return loss_v


def KL_Fisher(A, R, overreg=1.05):
    # A is bx3x3
    # R is bx3x3
    global _global_svd_fail_counter
    try:
        A, R = A.cpu(), R.cpu()
        U, S, V = torch.svd(A)
        with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            rotation_candidate = torch.matmul(U, V.transpose(1, 2))
            s3sign = torch.det(rotation_candidate)
        S_sign = S.clone()
        S_sign[:, 2] *= s3sign
        log_normalizer = logC_F(S_sign)
        log_exponent = -torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter - 1)
        return (log_exponent + overreg * log_normalizer).cuda()
    except RuntimeError as e:
        _global_svd_fail_counter += 10  # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100:  # we seem to have gotten these problems more often than 10% of batches
            for i in range(A.shape[0]):
                print(A[i])
            raise e
        else:
            print('SVD returned NAN fail counter = {}'.format(_global_svd_fail_counter))
            return None


def fisher_mode(A):
    A = A.view(-1, 3, 3).cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    U[:, :, 2] *= s3sign.view(-1, 1)
    R = torch.matmul(U, V.transpose(1, 2))
    R = R.cuda()
    return R

def _horner(arr, x):
    z = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(arr[0])
    for i in range(1, len(arr)):
        z.mul_(x).add_(arr[i])
    return z


torch_bessel0_a = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
torch_bessel0_b = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]


def bessel0(x):  # always supressed by exp(x)
    # x is of size (-1)
    abs_x = torch.abs(x)
    mask = abs_x <= 3.75
    e1 = _horner(torch_bessel0_a, (abs_x / 3.75) ** 2) / torch.exp(abs_x)
    e2 = _horner(torch_bessel0_b, 3.75 / abs_x) / torch.sqrt(abs_x)
    e2[mask] = e1[mask]
    return e2


def torch_integral(f, v, from_x, to_x, N):
    with torch.no_grad():
        # computes ret_i = \int_{from_x}^{to_x} f(x,v_i)
        # where N is number of trapezoids + 1 per v_i
        rangee = torch.arange(N, dtype=v.dtype, device=v.device)
        x = (rangee * ((to_x - from_x) / (N - 1)) + from_x).view(1, N)
        weights = torch.empty((1, N), dtype=v.dtype, device=v.device).fill_(1)
        weights[0, 0] = 1 / 2
        weights[0, -1] = 1 / 2
        y = f(x, v)
        return torch.sum(y * weights, dim=1) * (to_x - from_x) / (N - 1)


def integrand_CF(x, s):
    # x is (1, N)
    # s is (-1, 3)
    # return (-1, N)
    # s is sorted from large to small
    f1 = (s[:, 1] - s[:, 2]) / 2
    f2 = (s[:, 1] + s[:, 2]) / 2
    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (s[:, 2] + s[:, 0]).view(-1, 1) * (x - 1).view(1, -1)
    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = torch.exp(a3)
    ret = i1 * i2 * i3
    return ret


def integrand_Cdiff(x, s):
    s2 = s[:, 0]
    s1 = torch.max(s[:, 1:], dim=1).values
    s0 = torch.min(s[:, 1:], dim=1).values
    f1 = (s1 - s0) / 2
    f2 = (s1 + s0) / 2
    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (s0 + s2).view(-1, 1) * (x - 1).view(1, -1)
    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = x.view(1, -1)
    i4 = torch.exp(a3)
    return i1 * i2 * i3 * i4


class class_logC_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        N = 512
        # input is (..., 3) correspond to SINGULAR VALUES of F (NOT Lambda)
        shape = input.shape
        input_v = input.view(-1, 3)
        factor = 1 / 2 * torch_integral(integrand_CF, input_v, -1, 1, N)
        log_factor = torch.log(factor)
        log_supress = torch.sum(input_v, dim=1)
        ctx.save_for_backward(input, factor)
        return (log_factor + log_supress).view(shape[:-1])

    @staticmethod
    def backward(ctx, grad):
        S, factor = ctx.saved_tensors
        S = S.view(-1, 3)
        N = 512
        ret = torch.empty((S.shape[0], 3), dtype=S.dtype, device=S.device)
        for i in range(3):
            cv = torch.cat((S[:, i:], S[:, :i]), dim=1)
            ret[:, i] = 1 / 2 * torch_integral(integrand_Cdiff, cv, -1, 1, N)
        ret /= factor.view(-1, 1)
        ret *= grad.view(-1, 1)
        return ret.view((*grad.shape, 3))


logC_F = class_logC_F.apply