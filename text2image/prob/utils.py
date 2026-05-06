import json
from pathlib import Path
import torch
import os
import sys



_DEFAULT_IMAGEREWARD_DIR = Path("/root/autodl-tmp/ImageReward")
_DEFAULT_IMAGEREWARD_CKPT = _DEFAULT_IMAGEREWARD_DIR / "ImageReward.pt"
_DEFAULT_IMAGEREWARD_MED = _DEFAULT_IMAGEREWARD_DIR / "med_config.json"
_DEFAULT_VQASCORE_PROMPTS = Path("/root/diffusion_ranker/t2v_metrics-main/prompts/genai_image_1600.json")




def _load_prompts(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    # Format A: ["prompt 1", "prompt 2", ...] (your current COCO split files)
    if isinstance(obj, list):
        prompts = [str(x).strip() for x in obj if isinstance(x, str) and str(x).strip()]

    # Format B: {"annotations":[{"caption":"..."}, ...]} (raw COCO captions json)
    elif isinstance(obj, dict) and isinstance(obj.get("annotations"), list):
        prompts = [
            str(row.get("caption", "")).strip()
            for row in obj["annotations"]
            if isinstance(row, dict) and str(row.get("caption", "")).strip()
        ]
        
    # Format C: {"00000":{"prompt":"..."}, ...} (GenAI/VQAScore style)
    elif isinstance(obj, dict):
        prompts = [
            str(row.get("prompt", "")).strip()
            for _, row in sorted(obj.items())
            if isinstance(row, dict) and str(row.get("prompt", "")).strip()
        ]
    else:
        raise ValueError(f"Unsupported prompts json format: {type(obj)} in {path}")

    # Keep first occurrence order while removing duplicates.
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise ValueError(f"No valid prompt found in {path}")
    return prompts


def ensure_image_reward_on_path() -> None:
    """
    Add local ImageReward repo to sys.path so `import ImageReward` works.
    """
    repo_dir = Path("/root/diffusion_ranker/ImageReward-main").resolve()
    s = str(repo_dir)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_imagereward(
    device: torch.device,
    ckpt_path: Path = _DEFAULT_IMAGEREWARD_CKPT,
    med_config: Path = _DEFAULT_IMAGEREWARD_MED,
):
    ensure_image_reward_on_path()
    import ImageReward as RM
    from transformers import BertTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _ = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    _orig = BertTokenizer.from_pretrained

    def _offline_from_pretrained(*args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        return _orig(*args, **kwargs)

    BertTokenizer.from_pretrained = _offline_from_pretrained
    model = RM.load(str(ckpt_path), device=device, med_config=str(med_config))
    model.eval()
    return model
