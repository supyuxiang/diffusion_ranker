import torch


def _to_bchw_rgb_255(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim not in (3, 4):
        raise ValueError(f"Expected image ndim 3 or 4, got shape {tuple(image.shape)}")
    x = image.to(torch.float32)
    squeeze_back = x.ndim == 3
    if squeeze_back:
        if x.shape[0] != 3:
            raise ValueError(f"Expected RGB image with C=3, got shape {tuple(x.shape)}")
        x = x.unsqueeze(0)
    elif x.shape[1] != 3:
        raise ValueError(f"Expected RGB batch with C=3, got shape {tuple(x.shape)}")
    if x.max() <= 1.0:
        x = x * 255.0
    return x, squeeze_back


def brightness_reward(image: torch.Tensor) -> torch.Tensor:
    """Brightness reward: luma mean in 0-255 scale."""
    x, squeeze_back = _to_bchw_rgb_255(image)
    r, g, b = x[:, 0], x[:, 1], x[:, 2] # b,3,pixel,pixel -> b, pixel, pixel
    y = 0.299 * r + 0.587 * g + 0.114 * b
    out = y.mean(dim=(1, 2)) # b, pixel, pixel -> b = scale
    return out[0] if squeeze_back else out


def contrast_reward(image: torch.Tensor) -> torch.Tensor:
    """Contrast reward: luma standard deviation in 0-255 scale."""
    x, squeeze_back = _to_bchw_rgb_255(image)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    out = y.std(dim=(1, 2), unbiased=False)
    return out[0] if squeeze_back else out


def saturation_reward(image: torch.Tensor) -> torch.Tensor:
    """Saturation reward: HSV-S channel mean in 0-255 scale."""
    x, squeeze_back = _to_bchw_rgb_255(image)
    cmax = x.max(dim=1).values # b,3,pixel,pixel -> b, pixel, pixel
    cmin = x.min(dim=1).values # b,3,pixel,pixel -> b, pixel, pixel
    # HSV saturation in [0,1]: (cmax-cmin)/cmax, with 0 when cmax==0
    s01 = torch.where(cmax > 0, (cmax - cmin) / cmax, torch.zeros_like(cmax)) # b, pixel, pixel -> b, pixel, pixel
    out = (s01 * 255.0).mean(dim=(1, 2)) # b, pixel, pixel -> b = scale
    return out[0] if squeeze_back else out


def colorfulness_reward(image: torch.Tensor) -> torch.Tensor:
    """Colorfulness reward (Hasler & Susstrunk) per image."""
    x, squeeze_back = _to_bchw_rgb_255(image)
    r, g, b = x[:, 0], x[:, 1], x[:, 2] # b,3,pixel,pixel -> b, pixel, pixel
    rg = (r - g).abs()
    yb = ((r + g) * 0.5 - b).abs()
    std_rg = rg.std(dim=(1, 2), unbiased=False) # b, pixel, pixel -> b = scale
    std_yb = yb.std(dim=(1, 2), unbiased=False) # b, pixel, pixel -> b = scale
    mean_rg = rg.mean(dim=(1, 2)) # b, pixel, pixel -> b = scale
    mean_yb = yb.mean(dim=(1, 2)) # b, pixel, pixel -> b = scale
    out = torch.sqrt(std_rg.square() + std_yb.square()) + 0.3 * torch.sqrt(mean_rg.square() + mean_yb.square()) # b, pixel, pixel -> b = scale
    return out[0] if squeeze_back else out



