#!/usr/bin/env python3
"""
DiT Random Search + CLIPScore verifier（仅读预生成图）。init 固定 N_INIT=1024（与 FID 脚本一致）。

须先用 ``search/rollout.py`` 生成 ``candidate_0000`` …（``image_{idx}_{label}.png``）与 ``init1024/``。

每轮从 K 个 candidate 池的同一位置切片取 ``batch_size`` 张图，算 batch 平均 CLIP 图文余弦相似度，
选最高的一批；胜出路径记入 ``best_path_ls``，最后 ``shutil.copy2`` 到 ``best_images/``，文件名保留
``image_{序}_{class_id}.png``，供 ``compute_fid_is_clipscore`` 解析类别。

默认 search ``--batch_size=32``（小于 FID 版）。

**本地 CLIP 目录**：若为 HuggingFace 格式（含 ``config.json`` / ``preprocessor_config.json``），须用
``--clip_backend hf --clip_model_path <目录>``。OpenCLIP 后端仅支持：内置 ``pretrained`` 标签、
``.pt/.bin`` 权重文件，或含 ``open_clip_config.json`` 的 OpenCLIP 导出目录（``local-dir:``）。

用法::

  python3 search/dit_random_search_clipscore.py \
      --output_dir /root/autodl-tmp/eval_dit_rs_clip \
      --deno_steps 10 --guidance_scale 1.0 \
      --num_candidates 16 --batch_size 32

  # 本地 HF CLIP（如 openai_clip-vit-large-patch14）::
  #   --clip_backend hf --clip_model_path /root/autodl-tmp/openai_clip-vit-large-patch14
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dit_random_search_common import load_imagenet_simple_labels, prompt_for_class

from utils import compute_fid_is_clipscore, set_device, set_seed

N_INIT = 1024

_DEFAULT_HF_CLIP_ROOT = Path(
    os.environ.get(
        "CLIP_HF_CACHE",
        os.environ.get(
            "CLIP_MODEL_PATH",
            "/root/autodl-tmp/openai_clip-vit-large-patch14",
        ),
    )
)


def sorted_rollout_image_paths(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def class_id_from_rollout_path(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"expected image_*_*_label.png, got {path.name}")
    return int(parts[2])


def load_images_and_class_ids(paths: list[Path]) -> tuple[list[Image.Image], list[int]]:
    images: list[Image.Image] = []
    cids: list[int] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
        cids.append(class_id_from_rollout_path(p))
    return images, cids


def resolve_hf_clip_snapshot_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    ref = path / "refs" / "main"
    if ref.is_file():
        snap_id = ref.read_text(encoding="utf-8").strip()
        cand = path / "snapshots" / snap_id
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"无法在 {path} 解析 CLIP 权重（需要 config.json 或 refs/main + snapshots/<hash>）"
    )


def load_hf_clip(model_dir: Path, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to(device).float().eval()
    return model, processor


def load_open_clip(model_name: str, pretrained: str, device: str):
    """加载 OpenCLIP。

    ``pretrained`` 可以是：

    - 标签名（如 ``openai``、``laion400m_e32``），与 ``model_name``（如 ``ViT-L-14``）配合；
    - **文件**：``.pt`` / ``.bin`` 等，``create_model_and_transforms(model_name, pretrained=路径)``；
    - **目录**：仅当内含 ``open_clip_config.json`` 时为 OpenCLIP 包，使用 ``local-dir:`` 加载。

    HuggingFace Transformers 的 CLIP 目录（只有 ``config.json``、无 ``open_clip_config.json``）
    **不能**走本函数；请改用 ``load_hf_clip`` / ``--clip_backend hf``。
    """
    import open_clip
    from torchvision import transforms

    p = Path(pretrained).expanduser()
    model_id: str
    pretrained_arg: str | None

    if p.is_file():
        model_id = model_name
        pretrained_arg = str(p.resolve())
    elif p.is_dir():
        root = p.resolve()
        if (root / "open_clip_config.json").is_file():
            model_id = f"local-dir:{root}"
            pretrained_arg = None
        else:
            raise ValueError(
                f"OpenCLIP 无法从该目录加载（需要 open_clip_config.json）: {root}\n"
                f"若为 HuggingFace CLIP 快照，请使用: --clip_backend hf --clip_model_path {root}"
            )
    else:
        model_id = model_name
        pretrained_arg = pretrained

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id, pretrained=pretrained_arg
    )
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.to(device).float().eval()

    if preprocess is not None:
        transform = preprocess
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    return model, tokenizer, transform


def mean_batch_clipscore_hf(
    model,
    processor,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    prompts = [prompt_for_class(labels_1k, cid, prompt_template) for cid in class_ids]
    rgb = [img.convert("RGB") for img in images]
    inputs = processor(text=prompts, images=rgb, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item()) if sim.numel() else float("-inf")


def mean_batch_clipscore_open_clip(
    model,
    tokenizer,
    image_transform,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    prompts = [prompt_for_class(labels_1k, cid, prompt_template) for cid in class_ids]
    text_tokens = tokenizer(prompts).to(device)
    tensors = torch.stack(
        [image_transform(img.convert("RGB")) for img in images],
        dim=0,
    ).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item()) if sim.numel() else float("-inf")


def save_init_images(init_paths: list[Path], output_dir: Path) -> None:
    """``image_{i:05d}_{class_id:05d}.png``，与 rollout / ``compute_fid_is_clipscore`` 一致。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(init_paths):
        cid = class_id_from_rollout_path(path)
        dest = output_dir / f"image_{i:05d}_{cid:05d}.png"
        shutil.copy2(path, dest)


def save_best_images(best_path_ls: list[Path], output_dir: Path, *, start_idx: int = 0) -> None:
    for j, path in enumerate(best_path_ls):
        i = start_idx + j
        cid = class_id_from_rollout_path(path)
        dest = output_dir / f"image_{i:05d}_{cid:05d}.png"
        shutil.copy2(path, dest)


@torch.inference_mode()
def run_random_search_clipscore(
    clip_backend: str,
    clip_model,
    clip_aux,
    labels_1k: list[str],
    prompt_template: str,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    device_str: str,
    candidate_pathes_dict: dict[int, list[Path]],
    n_init: int,
) -> list[Path]:
    best_path_ls: list[Path] = []
    current_idx = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for c in range(num_candidates):
        if len(candidate_pathes_dict[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d}: need >= {num_samples} images, got {len(candidate_pathes_dict[c])}"
            )

    for _ in tqdm(range(num_rounds), desc="random search clipscore"):
        n_this_batch = min(batch_size, num_samples - current_idx)
        if n_this_batch <= 0:
            break

        best_score = float("-inf")
        best_candidate_idx = -1

        for candidate_idx, candidate_path_ls in candidate_pathes_dict.items():
            sub = candidate_path_ls[current_idx : current_idx + n_this_batch]
            batch_images, batch_classes = load_images_and_class_ids(sub)
            if clip_backend == "hf":
                score = mean_batch_clipscore_hf(
                    clip_model,
                    clip_aux,
                    device_str,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            else:
                tokenizer, image_transform = clip_aux
                score = mean_batch_clipscore_open_clip(
                    clip_model,
                    tokenizer,
                    image_transform,
                    device_str,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            if score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx

        assert best_candidate_idx != -1
        win_paths = candidate_pathes_dict[best_candidate_idx][
            current_idx : current_idx + n_this_batch
        ]
        best_path_ls.extend(win_paths)
        current_idx += n_this_batch
        if current_idx >= num_samples:
            break

    return best_path_ls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/root/autodl-tmp/random_search_clipscore_50k_cfg1.0_deno10")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="每轮每个 candidate 池切片长度（search batch，默认 32）",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--deno_steps", type=int, default=10, help="与 rollout 目录名一致")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--prompt_template",
        default="a photo of a {name}",
        help="{name}、{id} 见 prompt_for_class",
    )
    p.add_argument("--clip_backend", choices=("hf", "open_clip"), default="hf")
    p.add_argument(
        "--clip_model_path",
        type=str,
        default=str(_DEFAULT_HF_CLIP_ROOT),
        help="hf：HuggingFace CLIP 目录（含 config.json）。环境变量 CLIP_HF_CACHE / CLIP_MODEL_PATH 可覆盖默认。",
    )
    p.add_argument(
        "--clip_model",
        default="ViT-L-14",
        help="open_clip：架构名；hf 时忽略。",
    )
    p.add_argument(
        "--clip_pretrained",
        default="openai",
        help="open_clip：标签或 .pt/.bin 路径或 OpenCLIP 包目录；hf 时忽略。",
    )
    return p.parse_args()


def main() -> None:
    imagenet_val = Path("/root/autodl-tmp/imagenet/val")
    cache_name = "imagenet_val"

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rollout_root = Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{args.guidance_scale}_deno{args.deno_steps}"
    )
    candidate_dir_ls = [
        rollout_root / f"candidate_{i:04d}" for i in range(args.num_candidates)
    ]
    init_dir = rollout_root / "init1024"

    assert all(
        len(list(candidate_dir.glob("image_*_*.png"))) == args.num_samples
        for candidate_dir in candidate_dir_ls
    ) and len(list(init_dir.glob("image_*_*.png"))) == N_INIT, (
        f"expected each candidate to have {args.num_samples} images and init {N_INIT}; "
        f"check rollout_root={rollout_root}"
    )

    set_seed(args.seed)
    device = set_device()
    dstr = "cuda" if device.type == "cuda" else "cpu"

    labels_1k = load_imagenet_simple_labels()
    clip_dir_resolved: Path | None = None
    if args.clip_backend == "hf":
        clip_dir_resolved = resolve_hf_clip_snapshot_dir(Path(args.clip_model_path))
        print(f"load HF CLIP: {clip_dir_resolved}")
        clip_model, clip_aux = load_hf_clip(clip_dir_resolved, dstr)
    else:
        print(f"load OpenCLIP {args.clip_model}/{args.clip_pretrained}")
        clip_model, clip_tok, clip_tf = load_open_clip(
            args.clip_model, args.clip_pretrained, dstr
        )
        clip_aux = (clip_tok, clip_tf)

    candidate_pathes_dict: dict[int, list[Path]] = {}
    for i in range(args.num_candidates):
        candidate_pathes_dict[i] = sorted(
            candidate_dir_ls[i].glob("image_*_*.png"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

    init_path_ls = sorted(
        init_dir.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    best_dir = output_dir / "best_images"
    print(f"init {N_INIT} -> {best_dir}")
    save_init_images(init_path_ls, best_dir)

    print(
        f"random search (clipscore): root={rollout_root}, "
        f"num_candidates={args.num_candidates}, batch_size={args.batch_size}"
    )
    best_path_ls = run_random_search_clipscore(
        args.clip_backend,
        clip_model,
        clip_aux,
        labels_1k,
        args.prompt_template,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        dstr,
        candidate_pathes_dict,
        N_INIT,
    )
    save_best_images(best_path_ls, best_dir, start_idx=N_INIT)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "verifier": "clipscore",
                "imagenet_val": str(imagenet_val),
                "candidates_root": str(rollout_root),
                "init_dir": str(init_dir),
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "num_candidates": args.num_candidates,
                "n_init": N_INIT,
                "deno_steps": args.deno_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "ref_cache_name": cache_name,
                "prompt_template": args.prompt_template,
                "clip_backend": args.clip_backend,
                **(
                    {
                        "clip_model_path": args.clip_model_path,
                        "clip_model_path_resolved": str(clip_dir_resolved),
                    }
                    if args.clip_backend == "hf" and clip_dir_resolved is not None
                    else {
                        "clip_model": args.clip_model,
                        "clip_pretrained": args.clip_pretrained,
                    }
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nfinal FID / IS / CLIPScore")
    metrics = compute_fid_is_clipscore(best_dir, imagenet_val, device=device)
    print(
        f"   FID: {metrics['fid']:.2f}, IS: {metrics['is']:.4f} ± {metrics['is_std']:.4f}, "
        f"CLIPScore: {metrics['clipscore']:.4f}"
    )
    (best_dir / "final_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
