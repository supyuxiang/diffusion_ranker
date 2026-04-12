from __future__ import annotations

import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any



INCEPTION_WEIGHTS_PATH = (
    "/root/diffusion_ranker/third_party/torch_fidelity_weights/"
    "weights-inception-2015-12-05-6726825d.pth"
)


def get_inception_feature_extractor(device: str):
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=(device == "cuda"),
        verbose=False,
        feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
    )
    


def _torchmetrics_inception_module_for_layer(weights_path: str | None, layer: str | int):
    """Return ``NoTrainInceptionV3`` with local weights, or the string/int layer id for default download."""
    from torchmetrics.image.fid import NoTrainInceptionV3

    if weights_path:
        return NoTrainInceptionV3(
            "inception-v3-compat",
            [str(layer)],
            feature_extractor_weights_path=weights_path,
        )
    return layer



from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms


Constants_ls = {
    'layer_ratio_ls':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'middle_ratio_ls':[0.1,0.25,0.5,0.75,0.9,1.0], # 在 target_step = (ratio*total_steps)上抽取表征然后训ranker
    'model_name_ls':['dit-xl-2-256','google_ddpm-celebahq-256'], # uncond: DiT-XL-2-256; cond: google/ddpm-celebahq-256
    'cfg_ls':[1.0,1.25,1.5], # 对于imagenet
    'lr_ls':[5e-4,1e-4,5e-5,1e-5],
    'weight_decay_ls':[1e-4,1e-5,1e-6],
    'num_layers_ls':[1,2,3,4],
    'batch_size_ls':[16,32,64,128],
    'proj_dim_ls':[32,64,128,256,512],
    'num_epochs_ls':[1,3,5,10,30,50],
    "optimizer_ls": {
        "adamw": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
        "sgd": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "nesterov": True,
        },
    },
    "scheduler_ls": {
        "cosine": {
            "T_max": 10,
            "eta_min": 1e-4,
        },
        "cosine_with_restarts": {
            "T_0": 10,
            "T_mult": 2,
        },
        "step": {
            "step_size": 10,
            "gamma": 0.1,
        },
    },
    'ranker_type_ls':['listwise','pointwise'],
    'parallel_size_ls':[2,4,8,16],
}



def image_reward_repo_root() -> Path:
    """Directory that contains the ``ImageReward`` package (for ``import ImageReward``).

    Order: ``IMAGEREWARD_ROOT`` env → ``/root/ImageReward-main`` if it exists →
    ``<diffusion_ranker_repo>/../ImageReward-main`` (sibling of this package).
    """
    if os.environ.get("IMAGEREWARD_ROOT"):
        return Path(os.environ["IMAGEREWARD_ROOT"]).expanduser().resolve()
    preferred = Path("/root/ImageReward-main")
    if preferred.is_dir():
        return preferred
    return (Path(__file__).resolve().parent.parent / "ImageReward-main").resolve()


def ensure_image_reward_on_path() -> None:
    """Prepend :func:`image_reward_repo_root` to ``sys.path`` when that directory exists."""
    root = image_reward_repo_root()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


Constants_default = {
    'ratio':0.6,
    'model_name':'dit-xl-2-256',
    'cfg':1.0,
    'seed':42,
    'lr':1e-4,
    'weight_decay':1e-4,
    'num_layers':1,
    'batch_size':16,
    'proj_dim':64,
    'num_epochs':1,
    'optimizer':'adamw',
    'scheduler':'cosine',
    'ranker_type':'listwise',
    'parallel_size':4,
}


def set_seed(seed:int=42):
    import os
    import random
    import numpy as np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def refresh_memory():
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def set_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ratio2index(ratio:float,num_layers:int):
    return round(num_layers * ratio)

def get_num_layers(model_name: str) -> int:
    """DiT depth from ``transformer/config.json`` (same as ``len(pipe.transformer.transformer_blocks)`` when loaded)."""
    model_path = model_name2path(model_name)
    cfg_path = model_path / "transformer" / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Missing {cfg_path}; place the DiT checkpoint under {model_path} (e.g. huggingface-cli download)."
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "num_layers" in cfg:
        return int(cfg["num_layers"])
    if "depth" in cfg:
        return int(cfg["depth"])
    raise KeyError(f"Expected num_layers or depth in {cfg_path}, got keys: {list(cfg.keys())}")




############    load models    ############


def model_name2path(model_name:str) -> Path:
    from pathlib import Path
    assert model_name in ['dit-xl-2-256','google_ddpm-celebahq-256']
    if model_name == 'dit-xl-2-256':
        return Path("/autodl-tmp/DiT-XL-2-256")
    if model_name == 'google_ddpm-celebahq-256':
        return Path("/root/autodl-tmp/google_ddpm-celebahq-256")


def _dtype_str_to_torch(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d in ("fp32", "float32"):
        return torch.float32
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype {dtype!r}; use fp32|fp16|bf16")


def dit_loader(
    model_name: str | Path,
    device: torch.device,
    *,
    dtype: str = 'fp16',
) -> DiTPipeline:
    """Load DiT pipeline. Prefer ``dtype=`` over legacy ``fp16=`` (fp32 / fp16 / bf16)."""
    from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDPMPipeline
    model_path = model_name if isinstance(model_name, Path) else model_name2path(model_name)

    pipe = DiTPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=_dtype_str_to_torch(dtype),
        device=str(device),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    return pipe


def ddpm_loader(
    model_name: str | Path,
    device: torch.device,
    dtype:str='fp16',
) -> DDPMPipeline:
    from diffusers import DDPMPipeline
    model_path = model_name if isinstance(model_name,Path) else model_name2path(model_name)
    pipe = DDPMPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=_dtype_str_to_torch(dtype),
        device=str(device),
    )
    pipe = pipe.to(device)
    return pipe


def model_loader(model_name:str, device: torch.device, dtype:str='fp16') -> DiTPipeline | DDPMPipeline:
    if model_name == 'dit-xl-2-256':
        return dit_loader(model_name, device=device, dtype=dtype)
    if model_name == 'google_ddpm-celebahq-256':
        return ddpm_loader(model_name, device=device, dtype=dtype)
    raise ValueError(f"Unknown model name: {model_name}")


############    compute metrics    ############

def _load_images_from_dir(
    folder: Path | str,
    max_images: int | None = None,
    subset_seed: int = 0,
) -> torch.Tensor:
    """Return ``(N, 3, H, W)`` uint8 RGB."""
    import random as _random

    from torchvision.transforms import PILToTensor

    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )
    if max_images is not None and len(paths) > max_images:
        rng = _random.Random(subset_seed)
        paths = paths.copy()
        rng.shuffle(paths)
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images in {folder} (supported: {exts})")
    tensors: list[torch.Tensor] = []
    to_t = PILToTensor()
    for p in paths:
        im = Image.open(p).convert("RGB")
        t = to_t(im)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0)
    if not batch.dtype == torch.uint8:
        batch = batch.to(torch.uint8)
    return batch


def _remove_corrupted_images(gen_dir: str | Path) -> int:
    """移除无法用 PIL 打开的损坏图片，返回移除数量（与 torch_fidelity 加载方式一致）"""
    gen_dir = Path(gen_dir)
    removed = 0
    exts = (".png", ".jpg", ".jpeg", ".webp")
    for p in list(gen_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        try:
            with Image.open(p) as img:
                img.load()
                img.convert("RGB")
        except Exception:
            p.unlink()
            removed += 1
    return removed



def compute_fid_is_clipscore(
    gen_dir: str | Path,
    real_dir: str | Path = Path("/root/autodl-tmp/imagenet/val"),
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    """FID / IS / CLIPScore：与 ``_load_images_from_dir`` 相同排序读图，并从 ``image_*_{label}.png`` 解析类别。"""
    gen_dir = Path(gen_dir)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        p for p in gen_dir.iterdir() if p.suffix.lower() in exts and p.is_file()
    )
    labels_ls: list[int] = []
    for p in paths:
        parts = p.stem.split("_")
        labels_ls.append(int(parts[-1]))

    real_images = _load_images_from_dir(real_dir)
    gen_images = _load_images_from_dir(gen_dir)
    # check
    if gen_images.shape[0] != len(labels_ls):
        raise ValueError(
            f"image count mismatch: tensors {gen_images.shape[0]} vs parsed labels {len(labels_ls)}"
        )
    metrics: dict[str, float] = {}
    metrics["fid"] = float(compute_fid(real_images, gen_images))
    is_mean, _is_std = compute_is(gen_images)
    metrics["is"] = float(is_mean)
    metrics["is_std"] = float(_is_std)
    prompts = imagenet_label2text_for_clipscore(labels_ls)
    metrics["clipscore"] = float(compute_clipscore(gen_images, prompts, device=device))
    return metrics


def compute_fid(real_images:torch.Tensor, gen_images:torch.Tensor) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False) # 2048是默认的feature维度
    fid.set_dtype(torch.float64)
    fid.update(real_images,real=True)
    fid.update(gen_images,real=False)
    return float(fid.compute().item())


def compute_is(gen_images:torch.Tensor) -> tuple[float, float]:
    from torchmetrics.image.inception import InceptionScore
    is_model = InceptionScore(normalize=False, splits=5) # splits=5，得到std然后report在paper中。 mean +- std
    is_model.update(gen_images)
    is_mean, is_std = is_model.compute()
    return float(is_mean), float(is_std)



@lru_cache(maxsize=1)
def _imagenet_simple_labels_tuple() -> tuple[str, ...]:
    """1000 个类名字符串；下标与 DiT / PyTorch ImageNet（WNID 字典序）一致。仅读 json 文件。"""
    # 离线标签：先运行 ``python3 scripts/build_imagenet_simple_labels.py``（从 autodl-pub devkit 生成 json）
    _DEFAULT_IMAGENET_LABELS_JSON = Path("/root/diffusion_ranker/imagenet-simple-labels.json")

    path = _DEFAULT_IMAGENET_LABELS_JSON.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"ImageNet labels json not found: {path}. "
            "Run once (offline): python3 scripts/build_imagenet_simple_labels.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or len(labels) != 1000:
        raise ValueError(f"expected list of 1000 strings in {path}")
    return tuple(str(x) for x in labels)


def imagenet_simple_labels_list() -> list[str]:
    """1000 个类名（list），下标与 DiT / ImageNet 一致；数据来自仓库 ``imagenet-simple-labels.json``。"""
    return list(_imagenet_simple_labels_tuple())


def imagenet_label2text_for_clipscore(label_ls: list[int]) -> list[str]:
    """ImageNet 类索引 ``0 <= idx < 1000`` → 可读短名称（来自 :func:`imagenet_labels_json_path`）。"""
    assert isinstance(label_ls, list) and all(isinstance(label, int) and 0 <= label < 1000 for label in label_ls), f"label_ls must be a list of int in [0, 999], got {label_ls}"
    prompt_template = "a photo of a {label}"
    formatted_prompts = [
        prompt_template.replace('{label}', _imagenet_simple_labels_tuple()[label])
        for label in label_ls
    ]
    return formatted_prompts



def compute_clipscore(
    gen_images: torch.Tensor,
    prompts: list[str],
    *,
    clip_model_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> float:
    """Mean CLIPScore (0–100) between images under ``gen_dir`` and ``texts``.

    ``torchmetrics`` 要求 ``model_name_or_path`` 为 **str**（内部用 ``"openai" in path`` 选加载分支）；
    本地目录名需包含 ``openai``（如 ``.../openai_clip-vit-base-patch32``），否则会报
    ``Invalid model_name_or_path``。未提供 ``clip_model_dir`` 时用环境变量 ``CLIP_MODEL_DIR`` 或默认
    ``/autodl-tmp/openai_clip-vit-base-patch32``。
    """
    from torchmetrics.multimodal.clip_score import CLIPScore

    n = gen_images.shape[0]
    if n != len(prompts):
        raise ValueError(f"image count mismatch: tensors {n} vs prompts {len(prompts)}")


    root = clip_model_dir or os.environ.get("CLIP_MODEL_DIR", "/autodl-tmp/openai_clip-vit-base-patch32")
    clip_path = Path(root).expanduser().resolve()
    if not clip_path.is_dir():
        raise FileNotFoundError(
            f"Missing CLIP dir {clip_path}; set CLIP_MODEL_DIR or huggingface-cli download."
        )
    model_id = str(clip_path)
    if "openai" not in model_id:
        raise ValueError(
            "torchmetrics CLIPScore only accepts local paths whose string contains 'openai'; "
            f"got {model_id!r}. Use e.g. openai_clip-vit-base-patch32 or pass HF id via a symlink name."
        )

    if device is None:
        device = set_device()
    else:
        device = torch.device(device)
    metric = CLIPScore(model_name_or_path=model_id).to(device)
    gen_images = gen_images.to(device)
    metric.update(gen_images, prompts)
    return float(metric.compute().item()) # 0-100, higher is better


# do not implement now
def compute_KID(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.kid import KernelInceptionDistance
    pass

def compute_Precision(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.precision import Precision
    pass

def compute_Recall(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.recall import Recall
    pass
# 



###########  my pipeline    ############

def _prepare_pil_for_save(img: Image.Image, target_short: int) -> Image.Image:
    """Prepare PIL image for saving. Resize if necessary."""
    if img.size != (target_short, target_short):
        return img.resize((target_short, target_short), Image.Resampling.LANCZOS)
    return img # no resize


def _save_pil_task(path: Path, img: Image.Image, *, fmt: str, jpeg_quality: int, png_compress_level: int) -> None:
    if fmt == "jpeg":
        img.save(path, quality=jpeg_quality, subsampling=0)
    else:
        img.save(path, compress_level=png_compress_level)


_ROLLOUT_SAVE_WORKERS = 4


@torch.inference_mode()
def rollout_dit_with_steps(
    pipe: DiTPipeline,
    save_dir: Path,
    num_steps: int,
    num_samples: int,
    batch_size: int,
    device: torch.device | None = None,
    seed: int=42,
    guidance_scale: float = 1.0,
    *,
    jpeg_quality: int = 92,
    png_compress_level: int = 1,
    image_size: int = 256,
):
    """
    For each batch: call ``DiTPipeline.__call__`` with ``num_inference_steps=num_steps`` so the
    scheduler uses an independent full trajectory (same semantics as the former manual loop).

    This is not inherently faster than a hand-written loop (both step over timesteps in Python);
    speed comes from ``torch.compile``, fp16, and batch size. Using the official API keeps parity
    with diffusers and avoids duplicating CFG / sigma-split logic.
    """
    if device is None:
        device = set_device()
    else:
        device = torch.device(device)
    num_classes = 1000 # for imagenet dataset
    img_counter = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches),desc=f"steps={num_steps}",leave=False)
    for batch_idx in pbar:
        batch_num_samples = min(batch_size, num_samples - batch_idx * batch_size)
        batch_labels_ls = torch.randint(0, num_classes, (batch_num_samples,), device=device).tolist() # [batch_size]
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx) # build a generator for each batch

        out = pipe(
            class_labels=batch_labels_ls,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ) # 
        batch_images = out.images # [batch_size, 3, 256, 256]

        to_save: list[tuple[Path, Image.Image]] = []
        for label, img in zip(batch_labels_ls, batch_images):
            if img_counter >= num_samples:
                break
            img = _prepare_pil_for_save(img, image_size)
            to_save.append((save_dir / f'image_{img_counter:05d}_{label:05d}.png', img.copy())) # save as png, save label info for clipscore
            img_counter += 1

        with ThreadPoolExecutor(max_workers=_ROLLOUT_SAVE_WORKERS) as ex:
            futures = [
                ex.submit(
                    _save_pil_task,
                    path,
                    im,
                    fmt="png",
                    jpeg_quality=jpeg_quality,
                    png_compress_level=png_compress_level,
                )
                for path, im in to_save
            ]
            for fut in as_completed(futures):
                fut.result()
    gc.collect()
    torch.cuda.empty_cache()

    return img_counter






@torch.inference_mode()
def rollout_ddpm_with_steps(
    pipe,
    save_dir: Path,
    num_steps: int,
    num_samples: int,
    batch_size: int,
    device: torch.device | None = None,
    seed: int = 42,
    *,
    jpeg_quality: int = 92,
    png_compress_level: int = 1,
    image_size: int = 256,
) -> int:
    """无条件 DDPM：按 ``num_inference_steps`` 独立采样，保存 ``{idx:05d}.png``（与 ``compute_metrics`` 递归读图兼容）。"""
    device = device or set_device()
    save_dir.mkdir(parents=True, exist_ok=True)
    img_counter = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"ddpm steps={num_steps}", leave=False):
        batch_n = min(batch_size, num_samples - batch_idx * batch_size)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        out = pipe(
            batch_size=batch_n,
            generator=gen,
            num_inference_steps=num_steps,
            output_type="pil",
        )
        to_save: list[tuple[Path, Image.Image]] = []
        for img in out.images:
            if img_counter >= num_samples:
                break
            im = _prepare_pil_for_save(img.convert("RGB"), image_size)
            to_save.append((save_dir / f"{img_counter:05d}.png", im.copy()))
            img_counter += 1
        with ThreadPoolExecutor(max_workers=_ROLLOUT_SAVE_WORKERS) as ex:
            futures = [
                ex.submit(
                    _save_pil_task,
                    path,
                    im,
                    fmt="png",
                    jpeg_quality=jpeg_quality,
                    png_compress_level=png_compress_level,
                )
                for path, im in to_save
            ]
            for fut in as_completed(futures):
                fut.result()
    gc.collect()
    torch.cuda.empty_cache()
    return img_counter



### random search fid reward pipeline

import numpy as np

def fid_features_to_statistics(features: np.ndarray) -> dict:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {"mu": mu, "sigma": sigma}


def merge_fid_statistics(stats_a: dict, n_a: int, stats_b: dict, n_b: int) -> dict:
    if n_a == 0:
        return stats_b
    if n_b == 0:
        return stats_a

    mu_a, sigma_a = stats_a["mu"], stats_a["sigma"]
    mu_b, sigma_b = stats_b["mu"], stats_b["sigma"]
    n = n_a + n_b

    mu_merged = (n_a * mu_a + n_b * mu_b) / n
    sum_xx_a = ((n_a - 1) * sigma_a if n_a > 1 else 0) + n_a * np.outer(mu_a, mu_a)
    sum_xx_b = ((n_b - 1) * sigma_b if n_b > 1 else 0) + n_b * np.outer(mu_b, mu_b)
    sum_xx = sum_xx_a + sum_xx_b
    sigma_merged = (sum_xx - n * np.outer(mu_merged, mu_merged)) / (n - 1) if n > 1 else np.zeros_like(sigma_a)

    return {"mu": mu_merged, "sigma": sigma_merged}


def fid_statistics_to_distance(stat_gen: dict, stat_ref: dict) -> float:
    mu1, sigma1 = stat_gen["mu"], stat_gen["sigma"]
    mu2, sigma2 = stat_ref["mu"], stat_ref["sigma"]
    diff = mu1 - mu2
    evals = np.linalg.eigvals(sigma1 @ sigma2)
    evals = np.maximum(evals.real, 0.0)
    tr_covmean = np.sum(np.sqrt(evals.astype("float64") + 1e-14))
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)




def random_search_fid_reward(
    pipe,
    save_dir:Path | str,
    real_dir:str | Path,
    deno_step:int,
    num_gen:int=50000,
    parallel_size:int=4,
    batch_size:int=512,
    seed:int=42,
    cfg:float=1.0,
    *,
    init_samples:int=1024, #  paper setting
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    real_dir = Path(real_dir)

    # init samples
    



##########################################################



    
        











































def _iter_image_tensor_batches(
    img_dir: Path,
    batch_size: int = 64,
    max_samples: int | None = None,
    subset_seed: int = 0,
    device: torch.device = torch.device("cpu"),
):
    """Yield uint8 [B, 3, 299, 299] tensor batches from a directory of images.

    Images are resized to 299×299 (Inception V3 native resolution) so that the
    torchmetrics internal resize is a no-op and we can stack variable-size images.
    """
    import random as _random
    from torchvision.transforms.functional import pil_to_tensor, resize

    dev = torch.device(device)
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
            yield torch.stack(tensors).to(dev)








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






if __name__ == "__main__":
    pass

