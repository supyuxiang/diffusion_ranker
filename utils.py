from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# torch-fidelity InceptionV3 weights (same file as torch.hub would download from GitHub releases).
FIDELITY_INCEPTION_WEIGHTS_FILENAME = "weights-inception-2015-12-05-6726825d.pth"


def resolve_fidelity_inception_weights_path() -> str | None:
    """Resolve local path to torch-fidelity Inception weights to avoid ``torch.hub`` download.

    Order: env ``TORCH_FIDELITY_INCEPTION_WEIGHTS`` (must exist if set), then
    ``<this_repo>/third_party/torch_fidelity_weights/``, ``/root/autodl-pub/torch_fidelity_weights/``,
    ``~/.cache/torch/hub/checkpoints/``. Returns ``None`` if not found (online download may run).

    On a networked machine, fetch then ``scp`` to the server (filename
    ``weights-inception-2015-12-05-6726825d.pth`` from torch-fidelity GitHub releases v0.2.0).
    """
    env = os.environ.get("TORCH_FIDELITY_INCEPTION_WEIGHTS", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"TORCH_FIDELITY_INCEPTION_WEIGHTS={env!r} is not an existing file."
            )
        return str(p)

    roots = [
        Path(__file__).resolve().parent / "third_party" / "torch_fidelity_weights",
        Path("/root/autodl-pub/torch_fidelity_weights"),
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    ]
    for r in roots:
        cand = r / FIDELITY_INCEPTION_WEIGHTS_FILENAME
        if cand.is_file():
            return str(cand.resolve())
    return None


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


def dit_laoder(
    model_name: str | Path,
    device: str,
    *,
    dtype: str = 'fp16',
) -> DiTPipeline | DDPMPipeline:
    """Load DiT pipeline. Prefer ``dtype=`` over legacy ``fp16=`` (fp32 / fp16 / bf16)."""
    from diffusers import DiTPipeline, DPMSolverMultistepScheduler, DDPMPipeline
    model_path = model_name if isinstance(model_name, Path) else model_name2path(model_name)

    pipe = DiTPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=_dtype_str_to_torch(dtype),
        device=device,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    return pipe


def ddpm_loader(
    model_name: str | Path,
    device:str,
    dtype:str='fp16',
) -> DDPMPipeline:
    from diffusers import DDPMPipeline
    model_path = model_name if isinstance(model_name,Path) else model_name2path(model_name)
    pipe = DDPMPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=_dtype_str_to_torch(dtype),
        device=device,
    )
    pipe = pipe.to(device)
    return pipe


def model_loader(model_name:str, device:str, dtype:str='fp16') -> DiTPipeline | DDPMPipeline:
    if model_name == 'dit-xl-2-256':
        return load_dit(model_name, device, dtype=dtype)
    if model_name == 'google_ddpm-celebahq-256':
        return load_ddpm(model_name, device, dtype=dtype)
    raise ValueError(f"Unknown model name: {model_name}")

##########################################################


############    compute metrics    ############
def compute_FID(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance
    pass

def compute_IS(gen_dir:str | Path) -> float:
    from torchmetrics.image.inception import InceptionScore
    pass

def compute_CLIPScore(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.multimodal.clip_score import CLIPScore
    pass



def compute_KID(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.kid import KernelInceptionDistance
    pass

def compute_Precision(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.precision import Precision
    pass

def compute_Recall(gen_dir:str | Path, real_dir:str | Path) -> float:
    from torchmetrics.image.recall import Recall
    pass



##########################################################


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




def compute_fid_and_is(gen_dir: str | Path, real_dir: str | Path):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    real_images = _load_images_from_dir(real_dir)
    gen_images = _load_images_from_dir(gen_dir)
    fid_value = compute_fid(real_images, gen_images)
    is_value, is_std = compute_is(gen_images)
    return fid_value, is_value, is_std

def compute_fid(real_images:torch.Tensor, gen_images:torch.Tensor) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    fid.set_dtype(torch.float64)
    fid.update(real_images,real=True)
    fid.update(gen_images,real=False)
    return float(fid.compute().item())

def compute_is(gen_images:torch.Tensor) -> tuple[float, float]:
    from torchmetrics.image.inception import InceptionScore
    is_model = InceptionScore(normalize=False, splits=5)
    is_model.update(gen_images)
    is_mean, is_std = is_model.compute()
    return float(is_mean), float(is_std)



def compute_clipscore(gen_dir:str, texts:list[str]) -> float:
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision.transforms.functional import pil_to_tensor
    clip_path = Path("/autodl-tmp/openai_clip-vit-base-patch32")
    if not clip_path.is_dir():
        raise FileNotFoundError(f"Missing {clip_path}; place the CLIP checkpoint under {clip_path} (e.g. huggingface-cli download).")
    metrics = CLIPScore(model_name_or_path=clip_path)
    metrics.update(gen_images, texts)
    return float(metrics.compute().item())



###########  baseline verifier    ############



def baseline_fid_verifier(model_name:str, cfg:float, denoising_steps:int=10) -> float:
    from pprint import pprint
    from tqdm import tqdm
    parallel_size_ls = Constants_ls['parallel_size_ls']
    for parallel_size in parallel_size_ls:
        metrics = run(model_name, cfg, parallel_size)
        pprint(metrics)
        print('-'*100)
    
    def run(model_name:str,cfg:float, parallel_size:int) -> dict:
        seed = Constants_default['seed']
        set_seed(seed)
        device = set_device()
        pipe = load_dit(model_name, device, fp16=True)
        pipe.scheduler.set_timesteps(denoising_steps)
        pbar = tqdm(pipe.scheduler.timesteps, desc=f'{model_name}-{cfg}-{parallel_size}')
        

def baseline_clipscore_verifier(
    model_name:str,
    cfg:float,
    denoising_steps,
    parallel_size:int,
    ):
    pass      


##########################################################



    
        






























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
) -> dict[str, Any]:
    """
    计算生成图像质量指标：先从 ``gen_dir`` 读生成图、从 ``real_dir`` 读真实图。FID / IS / KID 使用 torchmetrics；
    Precision/Recall 仍通过 torch_fidelity（torchmetrics 暂无等效实现）。
    real_cache_name 保留以兼容旧接口，torchmetrics 路径下不使用缓存。
    """
    gen_dir = Path(gen_dir)
    real_dir = Path(real_dir)
    n_removed = _remove_corrupted_images(gen_dir)
    if n_removed > 0:
        print(f"  已移除 {n_removed} 张损坏图片")

    out: dict[str, Any] = {}
    _batch = 64

    if fid or isc or kid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        from torchmetrics.image.kid import KernelInceptionDistance

        inc_w = resolve_fidelity_inception_weights_path()
        fid_kw = (
            dict(feature_extractor_weights_path=inc_w) if inc_w else {}
        )
        fid_m = (
            FrechetInceptionDistance(feature=2048, normalize=False, **fid_kw).to(device)
            if fid
            else None
        )
        if fid_m is not None:
            fid_m.set_dtype(torch.float64)
        # IS 必须在 ImageNet logits 上算 softmax；用 2048 维 pool 特征会得到接近均匀的假分布，IS≈1。
        is_feat = _torchmetrics_inception_module_for_layer(inc_w, "logits_unbiased")
        is_m = (
            InceptionScore(feature=is_feat, normalize=False).to(device) if isc else None
        )
        kid_feat = _torchmetrics_inception_module_for_layer(inc_w, 2048)
        kid_m = (
            KernelInceptionDistance(
                feature=kid_feat, subset_size=100, normalize=False
            ).to(device)
            if kid
            else None
        )

        if fid_m is not None or kid_m is not None:
            for batch in _iter_image_tensor_batches(
                real_dir, _batch, real_max_samples, real_subset_seed, device
            ):
                if fid_m is not None:
                    fid_m.update(batch, real=True)
                if kid_m is not None:
                    kid_m.update(batch, real=True)

        for batch in _iter_image_tensor_batches(gen_dir, _batch, device=device):
            batch = batch.to(device)
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

        prc_kw: dict[str, Any] = {
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

    inc_w = resolve_fidelity_inception_weights_path()
    is_feat = _torchmetrics_inception_module_for_layer(inc_w, "logits_unbiased")
    metric = InceptionScore(feature=is_feat, normalize=False).to(device)
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

        inc_w = resolve_fidelity_inception_weights_path()
        fid_kw = dict(feature_extractor_weights_path=inc_w) if inc_w else {}
        fid_metric = FrechetInceptionDistance(feature=2048, **fid_kw).to(device)
        for batch in _iter_image_tensor_batches(real_a, 64, device=device):
            fid_metric.update(batch, real=True)
        for batch in _iter_image_tensor_batches(real_b, 64, device=device):
            fid_metric.update(batch, real=False)
        fid_val = float(fid_metric.compute())

        is_feat = _torchmetrics_inception_module_for_layer(inc_w, "logits_unbiased")
        is_metric = InceptionScore(feature=is_feat, normalize=False).to(device)
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

