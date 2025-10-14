# borrow from https://github.com/open-mmlab/mmrotate/blob/b030f38909fc431be7ecb90772ac30da9da29bcb/mmrotate/models/losses/gaussian_dist_loss_v1.py

import torch, math


def safe_logdet(M):
    M = M.contiguous().to(torch.float32)  # avoid AMP instability
    try:
        L = torch.linalg.cholesky(M)
        return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1, keepdim=True)
    except RuntimeError:
        # fallback if not SPD
        sign, logdet = torch.linalg.slogdet(M)
        if not torch.all(sign > 0):
            raise ValueError("Matrix not positive definite")
        return logdet.reshape(-1, 1)


def xy_wh_r_2_xy_sigma(xywhr):
    """
    Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center points, shape (N, 2)
        sigma (torch.Tensor): covariance matrices, shape (N, 2, 2)
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5

    xy = xywhr[..., :2]

    # Clamp width/height to avoid tiny covariances
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]

    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)

    # Diagonal covariance
    S = 0.5 * torch.diag_embed(wh)

    # Covariance in original orientation
    # with torch.cuda.amp.autocast(enabled=False):
    #    sigma = R.bmm(S.square()).bmm(R.transpose(1, 2))
    # sigma = sigma.reshape(_shape[:-1] + (2, 2))
    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def kld_loss_for_assigning(pred, target, fun='log1p', tau=1.0):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    with torch.cuda.amp.autocast(enabled=False):
        mu_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
        mu_t, sigma_t = xy_wh_r_2_xy_sigma(target)

        N, _ = mu_p.shape
        M, _ = mu_t.shape

        mu_p = mu_p[:, None, :].expand(N, M, 2).reshape(-1, 2)
        sigma_p = sigma_p[:, None, :, :].expand(N, M, 2, 2).reshape(-1, 2, 2).float()
        mu_t = mu_t[None, :, :].expand(N, M, 2).reshape(-1, 2)
        sigma_t = sigma_t[None, :, :, :].expand(N, M, 2, 2).reshape(-1, 2, 2).float()
        return_shape = [N, M]

        delta = (mu_p - mu_t).unsqueeze(-1)

        # Cholesky decomposition for sigma_t (stable inverse + logdet)
        L_t = torch.linalg.cholesky(sigma_t)
        sigma_t_inv = torch.cholesky_inverse(L_t)
        logdet_t = 2.0 * torch.log(torch.diagonal(L_t, dim1=-2, dim2=-1)).sum(-1, keepdim=True)

        # Cholesky logdet for sigma_p
        L_p = torch.linalg.cholesky(sigma_p)
        logdet_p = 2.0 * torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)).sum(-1, keepdim=True)

        # KL divergence terms
        term1 = delta.transpose(-1, -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)  # Mahalanobis
        term2 = torch.diagonal(sigma_t_inv.matmul(sigma_p), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) \
                + (logdet_t - logdet_p)
        

        dis = term1 + term2 - 2
        kl_dis = dis.clamp(min=1e-6).reshape(return_shape)

        # Final loss transform
        if fun == 'sqrt':
            kl_loss = 1 - 1 / (tau + torch.sqrt(kl_dis))
        else:
            kl_loss = 1 - 1 / (tau + torch.log1p(kl_dis))
        return kl_loss

def kld_loss(pred, target, fun='log1p', tau=1.0):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    with torch.cuda.amp.autocast(enabled=False):
        mu_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
        mu_t, sigma_t = xy_wh_r_2_xy_sigma(target)

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2).float()
        sigma_t = sigma_t.reshape(-1, 2, 2).float()

        delta = (mu_p - mu_t).unsqueeze(-1)

        # --- Cholesky inverse + logdet (with fallback) ---
        def logdet_and_inv(sigma):
            try:
                L = torch.linalg.cholesky(sigma)
                sigma_inv = torch.cholesky_inverse(L)
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1, keepdim=True)
                return sigma_inv, logdet
            except RuntimeError:
                # fallback if not SPD
                sign, logdet = torch.linalg.slogdet(sigma)
                if not torch.all(sign > 0):
                    raise ValueError("Non-SPD covariance encountered in kld_loss")
                sigma_inv = torch.linalg.inv(sigma)  # slower but safe fallback
                return sigma_inv, logdet.unsqueeze(-1)

        sigma_t_inv, logdet_t = logdet_and_inv(sigma_t)
        _, logdet_p = logdet_and_inv(sigma_p)

        # --- KL divergence terms ---
        term1 = delta.transpose(-1, -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)  # Mahalanobis
        term2 = torch.diagonal(sigma_t_inv.matmul(sigma_p), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) \
                + (logdet_t - logdet_p)

        dis = term1 + term2 - 2
        kl_dis = dis.clamp(min=1e-6)

        if fun == 'sqrt':
            kl_loss = 1 - 1 / (tau + torch.sqrt(kl_dis))
        else:
            kl_loss = 1 - 1 / (tau + torch.log1p(kl_dis))
        return kl_loss