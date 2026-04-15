# 训练数据 `repr.pt` 导出张量（`mean/`、`last/`）说明

由脚本 [`export_training_repr_tensors.py`](./export_training_repr_tensors.py) 从 rollout 生成的 `repr.pt` 写出若干 `.pt` 文件，便于按「与教师分同一顺序」直接索引表征。

## 生成方式

```bash
python3 scripts/export_training_repr_tensors.py \
  --image_dir /path/to/dit_training_data_... \
  --repr_pt /path/to/repr.pt \
  --reward_json /path/to/reward_ls.json
```

默认 `--output_root` 与 `--image_dir` 相同，即在其下创建子目录。

## 目录布局

在 `output_root`（通常即训练图目录）下：

| 路径 | 含义 |
|------|------|
| `mean/{repr_key}.pt` | 文件名与 `repr.pt` 内键一致，如 `s4_r0.037037.pt`；该键下 **mean pooling** 向量按图顺序堆成 `(N,H)` |
| `last/{repr_key}.pt` | 同上，**last token** 向量 |

- `repr_key` 形如 `s{step_index}_r{layer_ratio}`；**不再**用 `step_index/(T-1)` 作为文件名前缀（该比值仍可在 dict 的 `step_ratio` 字段查看）。
- 每个 repr 键对应 **一对** 文件（`mean` 与 `last` 各一）。

## `torch.load(...)` 得到的对象

每个文件是一个 **字典**（`dict`），训练或分析时一般只关心 `tensor` 与元数据字段。

### 核心字段

| 键 | 类型 | 含义 |
|----|------|------|
| `tensor` | `torch.FloatTensor`，形状 `(N, H)` | **第 `i` 行** = 第 `i` 张图在该层、该去噪步上的表征向量 |
| `n_images` | `int` | `N`，与目录下 `sorted(glob("*.png"))` 的数量一致 |
| `hidden_dim` | `int` | `H`，与 DiT hidden 维一致（如 XL 为 1152） |

### 与 `repr.pt` 中键的对应关系

| 键 | 类型 | 含义 |
|----|------|------|
| `repr_key` | `str` | 原始键，如 `s4_r0.592593`：`s` 后为 **去噪步索引**，`r` 后为 **层比例**（与 `DitRolloutReprCapture` 一致） |
| `step_index` | `int` | 去噪步索引（整数） |
| `layer_ratio` | `float` | 层比例 `r` |
| `step_ratio` | `float` | `step_index / max(T-1, 1)`，`T` 来自 `repr.pt` 的 `meta.num_inference_steps` |

### 行顺序（与 `reward_ls.json` 对齐）

| 键 | 类型 | 含义 |
|----|------|------|
| `row_order` | `str` | 固定说明：行顺序 = `sorted(image_dir.glob("*.png"))`，与 `ranker/labeling.py` 生成 `reward_ls.json` 时遍历图的顺序 **一致** |

因此：

- `tensor[i]` ↔ `reward_ls.json` 中第 `i` 个浮点数（同一套图、同一排序）。
- 与 `ranker/train.py` 里按 `repr.pt` 的 `image_save_names` 与扁平索引对齐样本的方式一致，此处显式按 **文件名排序** 铺成矩阵行。

### 溯源（可选）

| 键 | 含义 |
|----|------|
| `image_dir` | 训练图目录 |
| `repr_pt` | 源 `repr.pt` 路径 |
| `reward_json` | 校验用过的 `reward_ls.json` 路径 |

## 与原始 `repr.pt` 的关系

- 源 `repr.pt` 由 `rollout_trianing_data.py` 等写入：`batches[*].repr[repr_key]["mean"|"last"]` 形状为 `(batch_size, H)`。
- 导出脚本按 `image_save_names` 把每个文件名映射到 **排序后的全局行号**，再写入 `mean_buf` / `last_buf`，保证无重复、无遗漏（与 `*.png` 数量、`reward_ls` 长度一致）。

## 与 ranker 输入的区别

- **本导出**：每个文件一个 `(N, H)`，单 `(step, layer)`、单 pooling（mean 或 last）。
- **`ranker/train.py`**：通常把多个 `repr_key` 的 mean/last **拼成更长序列** `(N_tokens, H)` 再送入 encoder。若要用导出张量模拟某条训练样本，需按训练时 `selected_repr_keys` 顺序从多个文件中取行再拼接。
