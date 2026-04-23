from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms


__all__ = ["setup_seed",
 "compute_clip_score",
 "compute_fid_and_is",
 "compute_metrics",
 "compute_is_only",
 "prepare_real_images_dir",
 "prepare_real_images_from_hf",
 "eval_real_images",
 "load_dit",
]

def model_name2model(model_name):
    pass



def load_dit(dit_path:str | Path, device:str, fp16:bool=True):
    '''load DiT model from path'''
    from diffusers import DiTPipeline, DPMSolverMultistepScheduler
    pipe = DiTPipeline.from_pretrained(str(dit_path),
                                       local_files_only=True,
                                       torch_dtype=torch.float16 if fp16 else torch.float32,
                                       device=device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe



def setup_seed(seed: int, cuda_deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 种子。

    ``cuda_deterministic=True`` 时开启 cuDNN 确定性模式并尽量启用 PyTorch 确定性算法
   （更慢；个别算子可能仅告警或仍无法完全比特一致）。不需要时可传 ``False`` 保持原先速度。
    """
    import os
    import random

    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not cuda_deterministic:
        return

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 部分 CUDA / cuBLAS 在确定性 matmul 下需要（宜在进程早期设置）
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def compute_clip_score(images: list[Image.Image], device: str, prompt: str = "a face image") -> float:
    """Compute mean CLIPScore using torchmetrics (0-100 scale, higher is better)."""
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision.transforms.functional import pil_to_tensor

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)

    tensors = []
    for img in images:
        if isinstance(img, Image.Image):
            tensors.append(pil_to_tensor(img.convert("RGB")))
        else:
            raise ValueError(f"Invalid image type: {type(img)}")

    batch = torch.stack(tensors).to(device)
    texts = [prompt] * len(images)
    metric.update(batch, texts)
    return float(metric.compute().item())

def _remove_corrupted_images(gen_dir: str | Path) -> int:
    """移除无法用 PIL 打开的损坏图片，返回移除数量（与 torch_fidelity 加载方式一致）"""
    gen_dir = Path(gen_dir)
    removed = 0
    for p in list(gen_dir.glob("*.png")):
        try:
            with Image.open(p) as img:
                img.load()
                img.convert("RGB")
        except Exception:
            p.unlink()
            removed += 1
    return removed


def _iter_image_tensor_batches(
    img_dir: Path,
    batch_size: int = 64,
    max_samples: int | None = None,
    subset_seed: int = 0,
    device: str = "cpu",
):
    """Yield uint8 [B, 3, 299, 299] tensor batches from a directory of images.

    Images are resized to 299×299 (Inception V3 native resolution) so that the
    torchmetrics internal resize is a no-op and we can stack variable-size images.
    """
    import random as _random
    from torchvision.transforms.functional import pil_to_tensor, resize

    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)

    if max_samples is not None and len(paths) > max_samples:
        rng = _random.Random(subset_seed)
        paths = paths.copy()
        rng.shuffle(paths)
        paths = paths[:max_samples]

    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        tensors: list[torch.Tensor] = []
        for p in chunk:
            try:
                with Image.open(p) as img:
                    img.load()
                    t = pil_to_tensor(img.convert("RGB"))
                    if t.shape[-2] != 299 or t.shape[-1] != 299:
                        t = resize(t, [299, 299], antialias=True)
                    tensors.append(t)
            except Exception:
                continue
        if tensors:
            yield torch.stack(tensors).to(device)


def compute_metrics(
    gen_dir: str | Path,
    real_dir: str | Path,
    device: str,
    real_cache_name: str | None = None,
    fid: bool = True,
    isc: bool = True,
    kid: bool = False,
    prc: bool = False,
    real_max_samples: int | None = None,
    real_subset_seed: int = 0,
) -> dict:
    """
    计算生成图像质量指标。FID / IS / KID 使用 torchmetrics；
    Precision/Recall 仍通过 torch_fidelity（torchmetrics 暂无等效实现）。
    real_cache_name 保留以兼容旧接口，torchmetrics 路径下不使用缓存。
    """
    gen_dir = Path(gen_dir)
    real_dir = Path(real_dir)
    n_removed = _remove_corrupted_images(gen_dir)
    if n_removed > 0:
        print(f"  已移除 {n_removed} 张损坏图片")

    out: dict = {}
    _batch = 64

    if fid or isc or kid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        from torchmetrics.image.kid import KernelInceptionDistance

        fid_m = FrechetInceptionDistance(feature=2048).to(device) if fid else None
        # IS 必须在 ImageNet logits 上算 softmax；用 2048 维 pool 特征会得到接近均匀的假分布，IS≈1。
        is_m = InceptionScore(feature="logits_unbiased").to(device) if isc else None
        kid_m = KernelInceptionDistance(feature=2048, subset_size=100).to(device) if kid else None

        if fid_m is not None or kid_m is not None:
            for batch in _iter_image_tensor_batches(
                real_dir, _batch, real_max_samples, real_subset_seed, device
            ):
                if fid_m is not None:
                    fid_m.update(batch, real=True)
                if kid_m is not None:
                    kid_m.update(batch, real=True)

        for batch in _iter_image_tensor_batches(gen_dir, _batch, device=device):
            if fid_m is not None:
                fid_m.update(batch, real=False)
            if is_m is not None:
                is_m.update(batch)
            if kid_m is not None:
                kid_m.update(batch, real=False)

        if fid_m is not None:
            out["FID"] = float(fid_m.compute())
        if is_m is not None:
            is_mean, is_std = is_m.compute()
            out["IS"] = float(is_mean)
            out["IS_std"] = float(is_std)
        if kid_m is not None:
            kid_mean, kid_std = kid_m.compute()
            out["KID"] = float(kid_mean)
            out["KID_std"] = float(kid_std)

    if prc:
        import torch_fidelity

        prc_kw: dict = {
            "input2": str(real_dir),
            "input2_cache_name": real_cache_name or "real_ref",
        }
        prc_metrics = torch_fidelity.calculate_metrics(
            input1=str(gen_dir),
            cuda=(device == "cuda"),
            fid=False,
            isc=False,
            kid=False,
            prc=True,
            verbose=False,
            samples_find_deep=True,
            samples_resize_and_crop=256,
            **prc_kw,
        )
        out["Precision"] = prc_metrics.get("precision", float("nan"))
        out["Recall"] = prc_metrics.get("recall", float("nan"))
        out["F_score"] = prc_metrics.get("f_score", float("nan"))

    return out


def compute_fid_and_is(gen_dir: str | Path, real_dir: str | Path, device: str, real_cache_name: str | None = None):
    '''
    compute FID and IS of generated images and real images.
    torch_fidelity 只接受目录路径，不接受 PIL 列表。
    real_cache_name: 真实集缓存名，不同数据集用不同名避免混淆（如 celeb_hq_real, imagenet_val）
    计算前会移除无法识别的损坏图片，避免 PIL.UnidentifiedImageError。
    '''
    metrics = compute_metrics(gen_dir, real_dir, device, real_cache_name, fid=True, isc=True)
    return metrics.get("FID", float("nan")), metrics.get("IS", float("nan"))


def compute_is_only(gen_dir: str | Path, device: str) -> float:
    """Compute Inception Score only (no FID, no real images needed)."""
    from torchmetrics.image.inception import InceptionScore

    metric = InceptionScore(feature="logits_unbiased").to(device)
    for batch in _iter_image_tensor_batches(Path(gen_dir), 64, device=device):
        metric.update(batch)
    is_mean, _ = metric.compute()
    return float(is_mean)


def prepare_real_images_dir(dataset_path: str, output_dir: Path, num_samples: int = 50000) -> Path:
    """export CelebA-HQ dataset to image directory for FID calculation"""
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(ds))
    for i in tqdm(range(n), desc="export real images"):
        img = ds[i]["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(output_dir / f"{i:06d}.png")

    return output_dir


# huggan/CelebA-HQ 等旧 id 在 Hub 上常已失效；按顺序尝试直至成功。
_HF_CELEBAHQ_FALLBACK_IDS = (
    "nielsr/CelebA-HQ-resized-256",
    "Chris1/celebA-HQ",
    "KublaiKhan1/CelebA-HQ",
)


def _load_hf_image_split(hf_dataset_id: str, split: str):
    """Try load_dataset with fallbacks; returns (dataset_split, resolved_id)."""
    from datasets import load_dataset

    order: list[str] = []
    seen: set[str] = set()
    for d in (hf_dataset_id, *_HF_CELEBAHQ_FALLBACK_IDS):
        if d not in seen:
            seen.add(d)
            order.append(d)

    last_err: Exception | None = None
    for did in order:
        for trust_rc in (False, True):
            try:
                ds = load_dataset(did, split=split, trust_remote_code=trust_rc)
                if did != hf_dataset_id:
                    print(f"note: using Hub dataset {did!r} (requested {hf_dataset_id!r} was unavailable)")
                return ds, did
            except Exception as e:
                last_err = e
            try:
                raw = load_dataset(did, trust_remote_code=trust_rc)
                sp = split if split in raw else ("train" if "train" in raw else next(iter(raw.keys())))
                ds = raw[sp]
                if did != hf_dataset_id:
                    print(f"note: using Hub dataset {did!r} split={sp!r} (requested {hf_dataset_id!r} failed)")
                return ds, did
            except Exception as e:
                last_err = e
    raise RuntimeError(
        f"could not load any CelebA-HQ-style dataset (tried {order}); last error: {last_err!r}"
    ) from last_err


def prepare_real_images_from_hf(
    output_dir: Path,
    *,
    hf_dataset_id: str = "nielsr/CelebA-HQ-resized-256",
    split: str = "train",
    num_samples: int = 30000,
    shuffle_seed: int = 0,
) -> Path:
    """
    从 Hugging Face Hub 拉取人脸数据集并导出为 256×256 PNG，供 torch_fidelity 作真实参考集。
    若目录内已有不少于 num_samples 张 png 则跳过下载。首次拉取体积较大，需网络与 datasets 依赖。
    """
    import random

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.png"))
    if len(existing) >= num_samples:
        print(f"reuse HF-exported real images: {output_dir} ({len(existing)} files)")
        return output_dir

    for p in output_dir.glob("*.png"):
        p.unlink()

    print(
        f"loading HF dataset {hf_dataset_id!r} split={split!r} "
        f"(first run may download several GB; will try alternates if not found)..."
    )
    ds, resolved_id = _load_hf_image_split(hf_dataset_id, split)
    total = len(ds)
    n_export = min(num_samples, total)
    if n_export < num_samples:
        raise ValueError(
            f"{resolved_id} split={split} has only {total} samples, need at least {num_samples}"
        )

    row0 = ds[0]
    key = None
    for cand in ("image", "img"):
        if cand in row0:
            key = cand
            break
    if key is None:
        raise ValueError(
            f"could not find an image column in {resolved_id}; first row keys: {list(row0.keys())}"
        )

    idxs = list(range(total))
    rng = random.Random(shuffle_seed)
    rng.shuffle(idxs)
    idxs = idxs[:n_export]

    for i, ds_idx in enumerate(tqdm(idxs, desc="export HF real images")):
        raw = ds[ds_idx][key]
        if isinstance(raw, Image.Image):
            img = raw
        elif isinstance(raw, dict) and "bytes" in raw:
            import io

            img = Image.open(io.BytesIO(raw["bytes"]))
        else:
            import numpy as np

            img = Image.fromarray(np.asarray(raw))
        img = img.convert("RGB").resize((256, 256), Image.LANCZOS)
        img.save(output_dir / f"{i:06d}.png")

    print(f"exported {n_export} images to {output_dir}")
    return output_dir


def eval_real_images(args, output_dir: Path, device: str):
    """evaluate FID, CLIPScore, IS of real images (as baseline)"""
    from datasets import load_from_disk

    real_dataset = Path(args.real_dataset)
    assert real_dataset.exists(), f"dataset not found: {args.real_dataset}"

    num_samples = args.eval_real_samples
    real_export = output_dir / "real_images_export"
    existing = list(real_export.glob("*.png")) if real_export.exists() else []
    if len(existing) >= num_samples:
        print(f"reuse exported real images: {real_export} ({len(existing)} images)")
    else:
        print(f"exporting {num_samples} real images from {args.real_dataset} (only first time, then reuse)...")
        prepare_real_images_dir(args.real_dataset, real_export, num_samples=num_samples)

    images = []
    real_pngs = sorted(real_export.glob("*.png"))[:num_samples] if real_export.exists() else []
    if len(real_pngs) >= num_samples:
        for p in real_pngs[:num_samples]:
            images.append(Image.open(p).convert("RGB"))
        n = len(images)
    else:
        ds = load_from_disk(str(real_dataset))
        n = min(num_samples, len(ds))
        for i in range(n):
            img = ds[i]["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.resize((256, 256), Image.LANCZOS)
            images.append(img)

    print("\n--- real images metrics ---")

    half = n // 2
    real_a = output_dir / "real_split_a"
    real_b = output_dir / "real_split_b"
    real_a.mkdir(exist_ok=True)
    real_b.mkdir(exist_ok=True)
    for i in range(half):
        images[i].save(real_a / f"{i:04d}.png")
    for i in range(half, min(2 * half, n)):
        images[i].save(real_b / f"{i:04d}.png")

    fid_val = float("nan")
    is_val = float("nan")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore

        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        for batch in _iter_image_tensor_batches(real_a, 64, device=device):
            fid_metric.update(batch, real=True)
        for batch in _iter_image_tensor_batches(real_b, 64, device=device):
            fid_metric.update(batch, real=False)
        fid_val = float(fid_metric.compute())

        is_metric = InceptionScore(feature="logits_unbiased").to(device)
        for batch in _iter_image_tensor_batches(real_export, 64, device=device):
            is_metric.update(batch)
        is_mean, _ = is_metric.compute()
        is_val = float(is_mean)
    except Exception as e:
        print(f"  torchmetrics 计算失败: {e}")

    clip_val = compute_clip_score(images, device)

    print(f"  FID (self, 两半真实图): {fid_val:.2f}")
    print(f"  CLIPScore: {clip_val:.4f}")
    print(f"  IS: {is_val:.4f}")
    print("\n说明: FID 为同一数据集两半之间的值，应接近 0；CLIPScore/IS 为真实图像 baseline")

# def load_real_images(real_images_dir:Path) -> list[Image.Image]:
#     '''
#     load real images from directory
#     '''
#     real_images = []
#     for img_path in real_images_dir.glob("*.png"):
#         img = Image.open(img_path).convert("RGB")
#         real_images.append(img)
#     return real_images

if __name__ == "__main__":
    pass

