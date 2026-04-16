from __future__ import annotations

import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any




with open('/root/diffusion_ranker/imagenet-simple-labels.json', 'r') as f:
    labels_1k = json.load(f)



INCEPTION_WEIGHTS_PATH = (
    "/root/diffusion_ranker/third_party/torch_fidelity_weights/"
    "weights-inception-2015-12-05-6726825d.pth"
)

# HuggingFace 格式 CLIP（``compute_clipscore`` 默认 ``local_files_only`` 加载）
CLIP_HF_LOCAL_DIR = Path("/root/autodl-tmp/openai_clip-vit-large-patch14")

# ImageReward 源码根目录（``git clone`` 目标）；``import ImageReward`` 前需加入 ``sys.path``
IMAGE_REWARD_REPO_DIR = Path("/root/ImageReward-main")


def ensure_image_reward_on_path() -> None:
    s = str(IMAGE_REWARD_REPO_DIR.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def get_inception_feature_extractor(device: str):
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=(device == "cuda"),
        verbose=False,
        feature_extractor_weights_path=INCEPTION_WEIGHTS_PATH,
    )
    


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
    'parallel_size':16,
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
        return Path("/root/autodl-tmp/DiT-XL-2-256")
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



### model loader ########################################################
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


def model_loader(model_name:str, device: torch.device, dtype:str='bf16') -> DiTPipeline | DDPMPipeline:
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
    *,
    resize_crop: int | None = 256,
) -> torch.Tensor:
    """Return ``(N, 3, H, W)`` uint8 RGB.

    ``resize_crop``：若为正整数，先 ``Resize(s)`` + ``CenterCrop(s)`` 再堆叠，避免 ImageNet
    等目录里原始 JPEG 高宽不一导致 ``torch.stack`` 报错；与 ``compute_fid_is_from_dirs`` 的
    256 预处理一致。传 ``None`` 或 ``0`` 则保持原分辨率（须彼此尺寸相同才能 stack）。
    """
    import random as _random

    import torchvision.transforms as T
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
    if resize_crop is not None and resize_crop > 0:
        to_tensor = T.Compose(
            [T.Resize(resize_crop), T.CenterCrop(resize_crop), PILToTensor()]
        )
    else:
        to_tensor = PILToTensor()
    tensors: list[torch.Tensor] = []
    for p in paths:
        tensors.append(to_tensor(Image.open(p).convert("RGB")))
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


def _fidelity_images_dataset(
    directory: Path,
    *,
    resize_crop: int,
    find_deep: bool,
    exts: str,
    max_samples: int | None,
    subset_seed: int,
):
    """``torch_fidelity`` 风格：glob → 可选子采样 → Resize/CenterCrop → uint8 RGB tensor。"""
    import random as _random

    import torchvision.transforms as T
    from torch_fidelity.datasets import ImagesPathDataset, TransformPILtoRGBTensor
    from torch_fidelity.utils import glob_samples_paths

    directory = directory.expanduser().resolve()
    paths = glob_samples_paths(str(directory), find_deep, exts, None, verbose=False)
    if not paths:
        raise FileNotFoundError(f"No images in {directory} (exts={exts!r}, deep={find_deep})")
    if max_samples is not None and len(paths) > max_samples:
        rng = _random.Random(subset_seed)
        paths = paths.copy()
        rng.shuffle(paths)
        paths = paths[:max_samples]
    tfms: list = []
    if resize_crop > 0:
        tfms += [T.Resize(resize_crop), T.CenterCrop(resize_crop)]
    tfms.append(TransformPILtoRGBTensor())
    return ImagesPathDataset(paths, T.Compose(tfms))


def compute_fid_is_from_dirs(
    gen_dir: str | Path,
    real_dir: str | Path,
    device: str | torch.device,
    *,
    batch_size: int = 64,
    samples_resize_and_crop: int = 256,
    samples_find_deep: bool = True,
    samples_find_ext: str = "png,jpg,jpeg,JPEG,JPG",
    real_max_samples: int | None = None,
    gen_max_samples: int | None = None,
    real_subset_seed: int = 0,
    gen_subset_seed: int = 0,
    isc_splits: int = 5,
    verbose: bool = False,
    cache: bool = False,
) -> dict[str, float]:
    """目录级 FID / IS（``torch_fidelity.calculate_metrics``）。``input1``=生成集，``input2``=真实集。"""
    from torch_fidelity import KEY_METRIC_FID, KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD
    from torch_fidelity.metrics import calculate_metrics

    gen_dir = Path(gen_dir).expanduser().resolve()
    real_dir = Path(real_dir).expanduser().resolve()
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"gen_dir is not a directory: {gen_dir}")
    if not real_dir.is_dir():
        raise FileNotFoundError(f"real_dir is not a directory: {real_dir}")

    dev = torch.device(device) if isinstance(device, str) else device
    inc_p = Path(INCEPTION_WEIGHTS_PATH).resolve()

    ds_gen = _fidelity_images_dataset(
        gen_dir,
        resize_crop=samples_resize_and_crop,
        find_deep=samples_find_deep,
        exts=samples_find_ext,
        max_samples=gen_max_samples,
        subset_seed=gen_subset_seed,
    )
    ds_real = _fidelity_images_dataset(
        real_dir,
        resize_crop=samples_resize_and_crop,
        find_deep=samples_find_deep,
        exts=samples_find_ext,
        max_samples=real_max_samples,
        subset_seed=real_subset_seed,
    )

    raw = calculate_metrics(
        input1=ds_gen,
        input2=ds_real,
        cuda=(dev.type == "cuda"),
        batch_size=batch_size,
        fid=True,
        isc=True,
        kid=False,
        prc=False,
        verbose=verbose,
        cache=cache,
        feature_extractor_weights_path=str(inc_p),
        isc_splits=isc_splits,
        rng_seed=2020,
        samples_shuffle=True,
    )
    return {
        "FID": float(raw[KEY_METRIC_FID]),
        "IS": float(raw[KEY_METRIC_ISC_MEAN]),
        "IS_STD": float(raw[KEY_METRIC_ISC_STD]),
    }


def _inception_weights_kw() -> dict[str, str]:
    p = Path(INCEPTION_WEIGHTS_PATH)
    return dict(feature_extractor_weights_path=str(p.resolve())) if p.is_file() else {}


def _batched_inception_features(
    images: torch.Tensor,
    feat_extractor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """``images``: uint8 ``N×3×H×W``；返回 float32 CPU 特征矩阵。"""
    cuda = device.type == "cuda"
    feat_extractor.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            b = images[i : i + batch_size]
            if cuda:
                b = b.cuda(non_blocking=True)
            out = feat_extractor(b)
            if isinstance(out, tuple):
                out = out[0]
            chunks.append(out.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def compute_fid_is_clipscore(
    gen_dir: str | Path,
    real_dir: str | Path = Path("/root/autodl-tmp/imagenet/val"),
    *,
    device: torch.device | None = None,
    clip_model_dir: str | Path | None = None,
    clip_batch_size: int = 64,
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
    dev = device or set_device()
    metrics: dict[str, float] = {}
    metrics["fid"] = float(compute_fid(real_images, gen_images, device=dev))
    is_mean, _is_std = compute_is(gen_images, device=dev)
    metrics["is"] = float(is_mean)
    metrics["is_std"] = float(_is_std)
    prompts = imagenet_label2text_for_clipscore(labels_ls)
    metrics["clipscore"] = float(
        compute_clipscore(
            gen_images,
            prompts,
            device=device,
            clip_model_dir=clip_model_dir,
            clip_batch_size=clip_batch_size,
        )
    )
    return metrics



def compute_fid_is_clipscore_for_uncondition(
    gen_dir: str | Path,
    real_dir: str | Path,
    *,
    clip_model_dir: str | Path,
    clip_batch_size: int = 64,
) -> dict[str, float]:
    '''compute FID / IS / CLIPScore for uncondition image generation, typically for ddpm_unet'''

    gen_dir,real_dir = Path(gen_dir),Path(real_dir)
    assert gen_dir.is_dir() and real_dir.is_dir()

    real_images = _load_images_from_dir(real_dir)
    gen_images = _load_images_from_dir(gen_dir)
    device = set_device()

    prompt = 'a photo of a human face'
    prompts = [prompt] * gen_images.shape[0]

    metrics:dict[str,float] = {}

    metrics['fid'] = float(compute_fid(real_images,gen_images,device=device))
    is_mean, is_std = compute_is(gen_images,device=device)
    metrics['is'] = float(is_mean)
    metrics['is_std'] = float(is_std)
    metrics['clipscore'] = float(compute_clipscore(gen_images,prompts,device=device,clip_model_dir=clip_model_dir,clip_batch_size=clip_batch_size))
    return metrics





def compute_fid(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> float:
    """FID（``torch_fidelity``）：两批 uint8 ``N×3×H×W`` 图像，与 ``create_feature_extractor`` Inception 一致。"""
    from torch_fidelity import KEY_METRIC_FID
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.utils import create_feature_extractor

    dev = device or set_device()
    fe = create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=(dev.type == "cuda"),
        verbose=False,
        **_inception_weights_kw(),
    )
    r = _batched_inception_features(real_images, fe, dev, batch_size)
    g = _batched_inception_features(gen_images, fe, dev, batch_size)
    s_r = fid_features_to_statistics(r)
    s_g = fid_features_to_statistics(g)
    return float(fid_statistics_to_metric(s_g, s_r, False)[KEY_METRIC_FID])


def compute_is(
    gen_images: torch.Tensor,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
    isc_splits: int = 5,
) -> tuple[float, float]:
    """IS mean / std（``torch_fidelity.metric_isc``，与旧 torchmetrics ``splits=5`` 对齐）。"""
    from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD, isc_features_to_metric
    from torch_fidelity.utils import create_feature_extractor

    dev = device or set_device()
    fe = create_feature_extractor(
        "inception-v3-compat",
        ["logits_unbiased"],
        cuda=(dev.type == "cuda"),
        verbose=False,
        **_inception_weights_kw(),
    )
    logits = _batched_inception_features(gen_images, fe, dev, batch_size)
    out = isc_features_to_metric(
        logits, splits=isc_splits, shuffle=True, rng_seed=2020
    )
    return float(out[KEY_METRIC_ISC_MEAN]), float(out[KEY_METRIC_ISC_STD])



@lru_cache(maxsize=1)
def _imagenet_simple_labels_list() -> list[str]:
    """1000 个类名字符串；下标与 DiT / PyTorch ImageNet（WNID 字典序）一致。仅读 json 文件。"""
    # 离线标签：先运行 ``python3 scripts/build_imagenet_simple_labels.py``（从 autodl-pub devkit 生成 json）
    _DEFAULT_IMAGENET_LABELS_JSON = Path("/root/diffusion_ranker/imagenet-simple-labels.json")

    path = _DEFAULT_IMAGENET_LABELS_JSON.expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return list(str(x) for x in labels) # list


def imagenet_label2text_for_clipscore(label_ls: list[int]) -> list[str]:
    """ImageNet 类索引 ``0 <= idx < 1000`` → 可读短名称（来自 :func:`imagenet_labels_json_path`）。"""
    assert isinstance(label_ls, list) and all(isinstance(label, int) and 0 <= label < 1000 for label in label_ls), f"label_ls must be a list of int in [0, 999], got {label_ls}"
    prompt_template = "a photo of a {label}"
    formatted_prompts = [
        prompt_template.replace('{label}', _imagenet_simple_labels_list()[label])
        for label in label_ls
    ]
    return formatted_prompts



@torch.inference_mode()
def compute_clipscore(
    gen_images: torch.Tensor,
    prompts: list[str],
    *,
    clip_model_dir: str | Path | None = None,
    device: torch.device | str | None = None,
    clip_batch_size: int = 64,
) -> float:
    """Mean CLIPScore (0–100)，与 torchmetrics 定义一致：``max(100 * cos(E_img, E_text), 0)`` 再对样本平均。

    使用 **transformers** 从本地目录 ``local_files_only=True`` 加载。未传 ``clip_model_dir`` 时用
    常量 :data:`CLIP_HF_LOCAL_DIR`。
    """
    from transformers import CLIPModel, CLIPProcessor
    from torchvision.transforms.functional import to_pil_image

    n = gen_images.shape[0]
    assert gen_images.shape[0] == len(prompts)

    clip_path = (
        Path(clip_model_dir).expanduser().resolve()
        if clip_model_dir is not None
        else CLIP_HF_LOCAL_DIR.resolve()
    )
    
    device = set_device() if not device else torch.device(device)
    model = CLIPModel.from_pretrained(str(clip_path), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(clip_path), local_files_only=True)
    model = model.to(device).float().eval()

    total_score = 0.0
    total_n = 0
    for start in range(0, n, clip_batch_size):
        end = min(start + clip_batch_size, n)
        batch_imgs = gen_images[start:end]
        batch_prompts = prompts[start:end]
        pil_list = []
        for i in range(batch_imgs.shape[0]):
            t = batch_imgs[i]
            if t.device.type != "cpu":
                t = t.cpu()
            pil_list.append(to_pil_image(t))
        inputs = processor(
            text=batch_prompts, images=pil_list, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_embeds * text_embeds).sum(dim=-1)
        score = torch.clamp(sim * 100.0, min=0.0)
        total_score += float(score.sum().item())
        total_n += int(score.numel())

    return total_score / max(total_n, 1)


# # do not implement now
# def compute_KID(gen_dir:str | Path, real_dir:str | Path) -> float:
#     from torchmetrics.image.kid import KernelInceptionDistance
#     pass

# def compute_Precision(gen_dir:str | Path, real_dir:str | Path) -> float:
#     from torchmetrics.image.precision import Precision
#     pass

# def compute_Recall(gen_dir:str | Path, real_dir:str | Path) -> float:
#     from torchmetrics.image.recall import Recall
#     pass
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


class DitRolloutReprCapture:
    """在 ``DiTPipeline`` 单次 ``__call__`` 内，按去噪步与指定 DiT block 抓取 token 表征（mean / last）。"""

    def __init__(
        self,
        pipe: DiTPipeline,
        *,
        num_inference_steps: int,
        guidance_scale: float,
        layer_ratios: tuple[float, ...] = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ),
        step_ratio_ls:list[float]=[0.0, 0.25, 0.5, 0.75, 1.0],
        store_dtype: torch.dtype = torch.float16,
        rollout_base_seed: int | None = None,
    ) -> None:
        self.pipe = pipe
        self.T = num_inference_steps
        self.guidance_scale = guidance_scale
        self.store_dtype = store_dtype
        self.rollout_base_seed = rollout_base_seed
        self.step_ratio_ls = step_ratio_ls
        self.target_steps = sorted(
            {
                min(self.T - 1, max(0, round(ratio * (self.T - 1))))
                for ratio in self.step_ratio_ls
            }
        )
        n_blocks = len(pipe.transformer.transformer_blocks)
        self.layer_specs: list[tuple[float, int]] = []
        seen: set[int] = set()
        for r in layer_ratios:
            idx = int(round(r * (n_blocks - 1)))
            idx = max(0, min(n_blocks - 1, idx))
            if idx in seen:
                continue
            seen.add(idx)
            self.layer_specs.append((float(r), idx))
        self._handles: list = [] # 存储所有注册的 forward_hook
        self.records: list[dict] = [] # 存储所有捕获的表征
        self._capturing = False # 是否正在捕获表征
        self._step_counter = {"v": -1} # 存储当前步数
        self._current: dict[tuple[int, float], dict[str, torch.Tensor]] = {} # 存储当前捕获的表征
        self._batch_labels: list[int] = [] # 存储当前批次的标签
        self._batch_size = 0 # 存储当前批次的样本数量

    def _pre_transformer(self, _module, _inputs) -> None:
        if not self._capturing:
            return
        self._step_counter["v"] += 1

    def _make_block_hook(self, ratio: float):
        def hook(_module, _inp, out: torch.Tensor) -> None:
            if not self._capturing:
                return
            s = self._step_counter["v"]
            if s not in self.target_steps:
                return
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() != 3:
                return
            B = self._batch_size
            if self.guidance_scale > 1:
                h = h[:B]
            mean_p = h.mean(dim=1).cpu().to(self.store_dtype)
            last_p = h[:, -1, :].cpu().to(self.store_dtype)
            self._current[(int(s), float(ratio))] = {"mean": mean_p, "last": last_p}

        return hook

    def install(self) -> None: 
        # 注册 forward_pre_hook 和 forward_hook，用于捕获中间层的表征
        self._handles.append(
            self.pipe.transformer.register_forward_pre_hook(self._pre_transformer)
        )
        for ratio, idx in self.layer_specs: # 注册每个 block 的 forward_hook
            block = self.pipe.transformer.transformer_blocks[idx]
            self._handles.append(block.register_forward_hook(self._make_block_hook(ratio)))

    def remove(self) -> None:
        for h in self._handles: # 移除所有注册的 forward_hook
            h.remove()
        self._handles.clear()

    def set_capturing(self, active: bool) -> None:
        self._capturing = active

    def begin_batch(self, batch_labels: list[int]) -> None:
        self._step_counter["v"] = -1
        self._current = {} # 清空当前捕获的表征
        self._batch_labels = list(batch_labels) # 存储当前批次的标签
        self._batch_size = len(batch_labels) # 存储当前批次的样本数量

    def end_batch(
        self,
        batch_idx: int,
        image_index_start: int,
        *,
        generator_seed: int | None = None,
    ) -> None:
        if not self._capturing:
            return
        repr_dict = { # 存储当前捕获的表征
            f"s{s}_r{r:g}": v for (s, r), v in self._current.items()
        }
        gseed = generator_seed
        if gseed is None and self.rollout_base_seed is not None: # 如果 generator_seed 为空，则使用 rollout_base_seed + batch_idx
            gseed = self.rollout_base_seed + batch_idx
        image_save_names = [ # 存储当前批次的图像保存名称
            f"image_{image_index_start + i:05d}_{lab:05d}.png"
            for i, lab in enumerate(self._batch_labels)
        ]
        rec: dict = {
            "batch_idx": batch_idx,
            "image_index_start": image_index_start,
            "labels": self._batch_labels,
            "image_save_names": image_save_names,
            "repr": repr_dict,
        }
        if gseed is not None:
            rec["generator_seed"] = gseed
        self.records.append(rec)
        self._current.clear()

    def save_pt(self, path: Path) -> None: # 保存捕获的表征到 pt 文件
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "meta": {
                    "num_inference_steps": self.T,
                    "target_step_indices": self.target_steps,
                    "layer_ratios": [r for r, _ in self.layer_specs],
                    "layer_block_indices": [i for _, i in self.layer_specs],
                    "n_transformer_blocks": len(self.pipe.transformer.transformer_blocks),
                    "guidance_scale": self.guidance_scale,
                    "pooling": ("mean_dim1", "last_token_dim1"),
                    "store_dtype": str(self.store_dtype),
                    "rollout_base_seed": self.rollout_base_seed,
                    "batch_generator_seed_formula": "rollout_base_seed + batch_idx",
                    "image_filename_pattern": "image_{global_idx:05d}_{imagenet_label:05d}.png",
                },
                "batches": self.records,
            },
            path,
        )
        self.records.clear()


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
    repr_capture: DitRolloutReprCapture | None = None,
    repr_max_batches: int | None = None, # default is None, which means record all batches
):
    """
    For each batch: call ``DiTPipeline.__call__`` with ``num_inference_steps=num_steps`` so the
    scheduler uses an independent full trajectory (same semantics as the former manual loop).

    This is not inherently faster than a hand-written loop (both step over timesteps in Python);
    speed comes from ``torch.compile``, fp16, and batch size. Using the official API keeps parity
    with diffusers and avoids duplicating CFG / sigma-split logic.

    ``repr_capture``: 可选，在每次 ``pipe`` 内按步抓取若干 block 的 token 表征；``repr_max_batches``
    非空时只记录前若干 batch（仍生成全部图像）。
    """
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

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

        img_start = img_counter
        if repr_capture is not None:
            take_repr = repr_max_batches is None or batch_idx < repr_max_batches
            repr_capture.set_capturing(take_repr)
            if take_repr:
                repr_capture.begin_batch(batch_labels_ls)

        out = pipe(
            class_labels=batch_labels_ls,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ) # 

        if repr_capture is not None:
            repr_capture.end_batch(
                batch_idx, img_start, generator_seed=seed + batch_idx
            )
        batch_images = out.images # [batch_size, 3, 256, 256]

        to_save: list[tuple[Path, Image.Image]] = []
        for label, img in zip(batch_labels_ls, batch_images):
            if img_counter >= num_samples:
                break
            img = _prepare_pil_for_save(img, image_size)
            # very important to save label info for clipscore 
            save_path = save_dir / f'image_{img_counter:05d}_{label:05d}.png'
            to_save.append((save_path, img.copy())) # save as png, save label info for clipscore
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

        del out, gen, batch_images, to_save
        if repr_capture is not None and device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    gc.collect()
    torch.cuda.empty_cache()

    return img_counter

##############################################################
@torch.inference_mode()
def rollout_dit_perclass32_and_save_repr(
    *,
    layer_ratio: float,
    step_ratio: float,
    image_output_dir: Path | str,
    repr_output_pt: Path | str,
    model_name: str | Path = "dit-xl-2-256",
    num_inference_steps: int = 10,
    guidance_scale: float = 1.0,
    per_class: int = 64,
    batch_size: int = 256,
    seed: int = 42,
    dtype: str = "bf16",
    device: torch.device | str | None = None,
    image_size: int = 256,
    pooling_mode: str = "mean",
    mc: int = 8,
    save_mc_images: bool = False,
    store_dtype: torch.dtype = torch.float16,
    jpeg_quality: int = 92,
    png_compress_level: int = 1,
) -> torch.Tensor:
    """
    输入 ``layer_ratio`` + ``step_ratio``，生成 ``per_class x 1000`` 个 tensor 样本，
    并在对应 step 处做 MC 补全评估 reward。

    产物：
    - 图像：仅当 ``save_mc_images=True`` 时保存到
      ``image_output_dir/image_{global_idx:05d}_{label:05d}_mc{j:02d}.png``。
    - 表征+reward：``repr_output_pt``（同顺序 ``tensor``、``reward``）
    - 返回：``tensor``，shape = ``(1000 * per_class, hidden_dim)``
    """
    if not (0.0 <= float(layer_ratio) <= 1.0):
        raise ValueError(f"layer_ratio 应在 [0,1]，得到 {layer_ratio}")
    if not (0.0 <= float(step_ratio) <= 1.0):
        raise ValueError(f"step_ratio 应在 [0,1]，得到 {step_ratio}")
    if pooling_mode not in ("mean", "last"):
        raise ValueError(f"pooling_mode 仅支持 mean/last，得到 {pooling_mode!r}")
    if int(mc) <= 0:
        raise ValueError(f"mc 必须为正整数，得到 {mc}")
    if int(per_class) <= 0:
        raise ValueError(f"per_class 必须为正整数，得到 {per_class}")

    image_output_dir = Path(image_output_dir).expanduser().resolve()
    repr_output_pt = Path(repr_output_pt).expanduser().resolve()
    if save_mc_images:
        image_output_dir.mkdir(parents=True, exist_ok=True)
    repr_output_pt.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = set_device()
    else:
        device = torch.device(device)
    

    set_seed(seed)
    pipe = dit_loader(model_name, device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    # 强制离线，避免 ImageReward 初始化时 tokenizer 联网探测导致长时间无输出
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ensure_image_reward_on_path()
    try:
        from search.reward import load_imagereward
    except ModuleNotFoundError:
        # 兼容从 search/ 目录直接运行的旧入口
        from reward import load_imagereward
    rm_model = load_imagereward(device)

    num_classes = 1000
    per_class = int(per_class)
    num_samples = num_classes * per_class
    labels_all = [c for c in range(num_classes) for _ in range(per_class)] # c0,c0,c0,...,c999,c999,c999,...,c999
    num_batches = (num_samples + batch_size - 1) // batch_size

    # 创建一个 DitRolloutReprCapture 对象，用于捕获中间层的表征
    repr_capture = DitRolloutReprCapture(
        pipe,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        layer_ratios=(float(layer_ratio),),
        step_ratio_ls=[float(step_ratio)],
        store_dtype=store_dtype,
        rollout_base_seed=seed,
    )
    repr_capture.install()

    step_idx = min(
        int(num_inference_steps) - 1,
        max(0, round(float(step_ratio) * (int(num_inference_steps) - 1))),
    )
    # 先完成到 step_idx（包含该步）后，从后续 timesteps 做 MC 补全。
    k_completed = step_idx + 1
    repr_key = f"s{step_idx}_r{float(layer_ratio):g}"

    repr_chunks: list[torch.Tensor] = []
    reward_chunks: list[torch.Tensor] = []
    img_counter = 0

    def _broadcast_timesteps(t, latent_model_input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            is_mps = latent_model_input.device.type == "mps"
            is_npu = latent_model_input.device.type == "npu"
            if isinstance(t, float):
                dt = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dt = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([t], dtype=dt, device=latent_model_input.device)
        elif len(t.shape) == 0:
            timesteps = t[None].to(latent_model_input.device)
        else:
            timesteps = t
        return timesteps.expand(latent_model_input.shape[0])

    def _run_prefix_and_capture(
        batch_labels_ls: list[int],
        generator: torch.Generator,
        batch_idx: int,
        image_index_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        batch_num_samples = len(batch_labels_ls)
        latent_size = pipe.transformer.config.sample_size
        latent_channels = pipe.transformer.config.in_channels

        latents = torch.randn(
            (batch_num_samples, latent_channels, latent_size, latent_size),
            generator=generator,
            device=pipe._execution_device,
            dtype=pipe.transformer.dtype,
        )
        x = torch.cat([latents] * 2, dim=0) if guidance_scale > 1 else latents

        cls = torch.tensor(batch_labels_ls, device=pipe._execution_device).reshape(-1)
        cls_null = torch.tensor([1000] * batch_num_samples, device=pipe._execution_device)
        class_labels_input = torch.cat([cls, cls_null], 0) if guidance_scale > 1 else cls

        pipe.scheduler.set_timesteps(int(num_inference_steps))
        timesteps = pipe.scheduler.timesteps
        k = min(int(k_completed), len(timesteps))

        repr_capture.set_capturing(True)
        repr_capture.begin_batch(batch_labels_ls)
        for t in timesteps[:k]:
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)
            x = pipe.scheduler.scale_model_input(x, t)

            timesteps_b = _broadcast_timesteps(t, x)
            noise_pred = pipe.transformer(
                x, timestep=timesteps_b, class_labels=class_labels_input
            ).sample
            if guidance_scale > 1:
                eps, tail = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise_pred = torch.cat([eps, tail], dim=1)
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred
            x = pipe.scheduler.step(model_output, t, x, generator=generator).prev_sample

        repr_capture.end_batch(
            batch_idx, image_index_start, generator_seed=seed + batch_idx
        )
        rec = repr_capture.records.pop()
        repr_dict = rec["repr"]
        if repr_key not in repr_dict:
            raise KeyError(
                f"未抓到 repr_key={repr_key!r}；当前 keys={list(repr_dict.keys())[:8]}"
            )
        repr_batch = repr_dict[repr_key][pooling_mode].clone()
        # DPMSolverMultistepScheduler 内部有 step_index / model_outputs 状态；
        # 后续每个 MC 分支都应从同一 prefix 终态独立继续，避免状态串扰导致越界。
        import copy
        scheduler_after_prefix = copy.deepcopy(pipe.scheduler)
        return x, class_labels_input, latent_channels, timesteps, repr_batch, scheduler_after_prefix

    def _decode_suffix_mc_batch(
        x_start: torch.Tensor,
        class_labels_input: torch.Tensor,
        latent_channels: int,
        timesteps: torch.Tensor,
        scheduler_after_prefix,
        mc_index: int,
        batch_idx: int,
    ) -> list[Image.Image]:
        import copy
        scheduler_mc = copy.deepcopy(scheduler_after_prefix)
        gen_mc = torch.Generator(device=x_start.device).manual_seed(
            int(seed) + 10_000 + batch_idx * 9_973 + mc_index * 67
        )
        x = x_start.clone()
        for t in timesteps[k_completed:]:
            if guidance_scale > 1:
                half = x[: len(x) // 2]
                x = torch.cat([half, half], dim=0)
            x = scheduler_mc.scale_model_input(x, t)
            timesteps_b = _broadcast_timesteps(t, x)
            noise_pred = pipe.transformer(
                x, timestep=timesteps_b, class_labels=class_labels_input
            ).sample
            if guidance_scale > 1:
                eps, tail = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
                noise_pred = torch.cat([eps, tail], dim=1)
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred
            variance_noise = torch.randn(
                x.shape,
                generator=gen_mc,
                device=x.device,
                dtype=x.dtype,
            )
            x = scheduler_mc.step(
                model_output, t, x, generator=gen_mc, variance_noise=variance_noise
            ).prev_sample

        if guidance_scale > 1:
            latents, _ = x.chunk(2, dim=0)
        else:
            latents = x
        latents = 1 / pipe.vae.config.scaling_factor * latents
        samples = pipe.vae.decode(latents).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)

        out_images: list[Image.Image] = []
        for bi in range(samples.shape[0]):
            arr = samples[bi].detach().cpu().permute(1, 2, 0).float().numpy()
            out_images.append(
                Image.fromarray((arr * 255.0).round().clip(0, 255).astype("uint8"))
            )
        return out_images

    try:
        pbar = tqdm(
            range(num_batches),
            desc=f"perclass32+mc{mc} s={step_idx} r={float(layer_ratio):g}",
            leave=False,
        )
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_labels_ls = labels_all[start:end]
            batch_prompts = [f"a photo of a {labels_1k[int(l)]}" for l in batch_labels_ls]
            gen = torch.Generator(device=device).manual_seed(seed + batch_idx)

            x_mid, class_input, latent_c, timesteps, repr_batch, scheduler_after_prefix = _run_prefix_and_capture(
                batch_labels_ls=batch_labels_ls,
                generator=gen,
                batch_idx=batch_idx,
                image_index_start=img_counter,
            )
            repr_chunks.append(repr_batch)
            batch_n = len(batch_labels_ls)
            reward_sum = torch.zeros(batch_n, dtype=torch.float32)

            for mc_i in range(int(mc)):
                print(
                    f"[case l{float(layer_ratio):g}_s{step_idx}] "
                    f"batch {batch_idx + 1}/{num_batches} "
                    f"mc {mc_i + 1}/{int(mc)} running..."
                )
                batch_images = _decode_suffix_mc_batch(
                    x_start=x_mid,
                    class_labels_input=class_input,
                    latent_channels=latent_c,
                    timesteps=timesteps,
                    scheduler_after_prefix=scheduler_after_prefix,
                    mc_index=mc_i,
                    batch_idx=batch_idx,
                )
                if save_mc_images:
                    to_save: list[tuple[Path, Image.Image]] = []
                    for i, (label, img) in enumerate(zip(batch_labels_ls, batch_images)):
                        img = _prepare_pil_for_save(img, image_size)
                        save_path = image_output_dir / (
                            f"image_{(img_counter + i):05d}_{int(label):05d}_mc{mc_i:02d}.png"
                        )
                        to_save.append((save_path, img.copy()))

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
                # 图像已落盘后再做 ImageReward 打分
                for i, (prompt, img) in enumerate(zip(batch_prompts, batch_images)):
                    reward_sum[i] += float(rm_model.score(prompt, img))
                print(
                    f"[case l{float(layer_ratio):g}_s{step_idx}] "
                    f"batch {batch_idx + 1}/{num_batches} "
                    f"mc {mc_i + 1}/{int(mc)} done."
                )

                if save_mc_images:
                    del to_save
                del batch_images
            reward_chunks.append(reward_sum / float(mc))
            img_counter += batch_n
            batch_reward_mean = float((reward_sum / float(mc)).mean().item())
            case_tag = f"l{float(layer_ratio):g}_s{step_idx}"
            print(
                f"[case {case_tag}] batch {batch_idx + 1}/{num_batches} "
                f"saved={img_counter}/{num_samples} "
                f"batch_reward_mean={batch_reward_mean:.4f}"
            )
            del gen, x_mid, class_input, timesteps, repr_batch, scheduler_after_prefix, reward_sum
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        repr_capture.remove()

    if len(repr_chunks) != num_batches:
        raise RuntimeError(f"repr chunks 数 {len(repr_chunks)} 与批次数 {num_batches} 不一致")
    # 按全局样本顺序拼成单个大 tensor（不是按 batch 保存）
    repr_tensor = torch.cat(repr_chunks, dim=0).contiguous()
    reward_tensor = torch.cat(reward_chunks, dim=0).contiguous()
    if not isinstance(repr_tensor, torch.Tensor) or not isinstance(reward_tensor, torch.Tensor):
        raise RuntimeError("repr/reward 必须是 torch.Tensor")
    if repr_tensor.dim() != 2:
        raise RuntimeError(f"repr_tensor 期望 2D，得到 {tuple(repr_tensor.shape)}")
    if reward_tensor.dim() != 1:
        raise RuntimeError(f"reward_tensor 期望 1D，得到 {tuple(reward_tensor.shape)}")
    if repr_tensor.shape[0] != num_samples:
        raise RuntimeError(f"repr tensor 行数 {repr_tensor.shape[0]} != 期望 {num_samples}")
    if reward_tensor.shape[0] != num_samples:
        raise RuntimeError(f"reward tensor 行数 {reward_tensor.shape[0]} != 期望 {num_samples}")
    if save_mc_images and img_counter != num_samples:
        raise RuntimeError(f"保存图像数 {img_counter} != 期望 {num_samples}")

    torch.save(
        {
            "tensor": repr_tensor,
            "reward": reward_tensor,
            "n_images": int(repr_tensor.shape[0]),
            "hidden_dim": int(repr_tensor.shape[1]),
            "tensor_shape": list(repr_tensor.shape),
            "reward_shape": list(reward_tensor.shape),
            "repr_key": repr_key,
            "step_ratio": float(step_ratio),
            "step_index": int(step_idx),
            "layer_ratio": float(layer_ratio),
            "pooling_mode": pooling_mode,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "mc": int(mc),
            "save_mc_images": bool(save_mc_images),
            "seed": int(seed),
            "dtype": str(dtype),
            "image_dir": str(image_output_dir),
            "image_filename_pattern": "image_{global_idx:05d}_{imagenet_label:05d}_mc{mc_idx:02d}.png",
            "label_order": f"class-major, each class repeated {per_class} times",
        },
        repr_output_pt,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return repr_tensor


##############################################################



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
    """无条件 DDPM：按 ``num_inference_steps`` 独立采样，保存 ``{idx:05d}.png``（供 ``compute_fid_is_from_dirs`` 等读目录）。"""
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
            im = _prepare_pil_for_save(img.convert("RGB"), image_size) # resize if needed
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


    

##########################################################



    
        














































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

