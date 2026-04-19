import shutil
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))






def _sorted_candidate_images(candidate_dir: Path) -> list[Path]:
    return sorted(candidate_dir.glob("image_*_*.png"), key=lambda p: int(p.stem.split("_")[1]))


def _reset_best_dir(best_dir: Path) -> None:
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)


def save_best_images(best_path_ls: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in best_path_ls:
        copy_path = output_dir / path.name
        shutil.copy2(path,copy_path)
    print(f"saved {len(best_path_ls)} best images to {output_dir}")
