# Text-to-Image 详细执行清单（基于当前 diffusion_ranker 仓库）

本文档目标：把你现在的 class-conditional 基础扩展为可复现实验的 text-to-image 管线，并支持  
`Base / Best-of-N / FK Steering / Ranker` 的系统对比。

---

## 0. 项目目标与验收

- [ ] **目标定义**：在相同计算预算下，提升 text-image 对齐与人偏好，同时控制 FID/IS 不退化过多。
- [ ] **主结果**：完成 `GenEval` 上的主表（Base、CLIP、ImageReward、Ranker、FK variants）。
- [ ] **补充结果**：完成 `DrawBench`（质性+定量）与 `HPSv2`（人偏好导向）结果。
- [ ] **可复现性**：提供统一脚本，支持一键复现实验（固定 seeds、固定预算、固定模型版本）。

验收标准：
- [ ] 同一 prompt 集合上全部方法均可跑通并保存统一格式结果。
- [ ] 至少 3 次独立种子重复，结果可汇总 mean/std。
- [ ] 关键表格与曲线可自动生成（csv + plotting script）。

---

## 1. 基础目录与文件规范

建议在 `text2image/` 下建立如下结构：

- [ ] `text2image/data/`
  - [ ] `prompts/`（benchmark prompt 列表与元数据）
  - [ ] `prompt_repr/`（文本编码或 LLM hidden states）
- [ ] `text2image/generate/`
  - [ ] `build_candidate_pool.py`（离线候选池生成）
  - [ ] `build_mid_repr_pool.py`（中间步表示提取，可选）
- [ ] `text2image/reward/`
  - [ ] `score_clip.py`
  - [ ] `score_imagereward.py`
  - [ ] `score_pickscore_or_hpsv2.py`（二选一）
- [ ] `text2image/ranker/`
  - [ ] `build_ranker_trainset.py`
  - [ ] `train_ranker.py`
  - [ ] `eval_ranker.py`
- [ ] `text2image/search/`
  - [ ] `best_of_n.py`
  - [ ] `fk_steering.py`
- [ ] `text2image/eval/`
  - [ ] `run_geneval.py`
  - [ ] `run_drawbench.py`
  - [ ] `aggregate_metrics.py`
  - [ ] `plot_curves.py`
- [ ] `text2image/scripts/`
  - [ ] `run_all_baselines.sh`
  - [ ] `run_all_fk.sh`
  - [ ] `run_all_ranker.sh`

---

## 2. Benchmark 选择与数据准备

### 2.1 必选：GenEval
- [ ] 下载/整理 GenEval prompt 列表。
- [ ] 保存为统一格式（建议 jsonl）：
  - [ ] `prompt_id`
  - [ ] `prompt_text`
  - [ ] `benchmark="geneval"`
  - [ ] `tags`（count/color/position 等）
- [ ] 写 `prompt_loader`，支持按子任务筛选（ablation 需要）。

### 2.2 建议：DrawBench
- [ ] 整理 DrawBench prompt。
- [ ] 标记 prompt 类型（compositional/attribute/text rendering 等）。

### 2.3 建议：HPSv2
- [ ] 准备 HPSv2 评测需要的 prompt 子集或评分脚本。
- [ ] 与主 benchmark prompt 对齐（避免 apples-to-oranges）。

---

## 3. 候选池生成（离线）

目标：每个 prompt 先生成固定数量候选（例如 N_pool=64 或 128），供后续 `best-of-N` 和 `FK` 统一复用。

- [ ] 统一生成参数：
  - [ ] model ckpt/version
  - [ ] sampler/scheduler
  - [ ] denoise steps
  - [ ] guidance scale
  - [ ] image size
  - [ ] seed policy（全方法共享）
- [ ] 文件组织建议：
  - [ ] `outputs/candidate_pool/{benchmark}/{prompt_id}/images/*.png`
  - [ ] `outputs/candidate_pool/{benchmark}/{prompt_id}/meta.pt`（seed、step、cfg、latent id）
  - [ ] `outputs/candidate_pool/{benchmark}/{prompt_id}/repr.pt`（可选，供 ranker）
- [ ] 完整性检查脚本：
  - [ ] 每个 `prompt_id` 图片数量一致
  - [ ] 无损坏文件
  - [ ] meta 与图片顺序严格一一对应

---

## 4. Reward 模型接入（teacher + baseline）

### 4.1 CLIPScore
- [ ] 实现 batched 推理（避免重复 load 权重）。
- [ ] 统一 prompt template（直接用原始 prompt，不再 class label 模板）。
- [ ] 缓存 text embeddings（同 prompt 只编码一次）。

### 4.2 ImageReward
- [ ] 强制离线加载逻辑（`HF_HUB_OFFLINE=1` 等）。
- [ ] batched score，记录异常样本与重试机制。
- [ ] 保存原始分数与归一化分数（便于跨 prompt 比较）。

### 4.3 PickScore 或 HPSv2（二选一）
- [ ] 若追求人偏好可信度优先，选 HPSv2。
- [ ] 若追求工程易用，选 PickScore。
- [ ] 与 CLIP/IR 同格式输出：`scores/{reward_name}/{prompt_id}.pt`

---

## 5. Ranker 数据构建

### 5.1 final-step ranker（先做）
- [ ] 输入：candidate 图像对应的最终层/关键层表示 + prompt 表示。
- [ ] 监督：ImageReward（主）或 PickScore/HPSv2（辅）。
- [ ] 保存训练样本：
  - [ ] `feature_tensor`（大 tensor，不分散碎片）
  - [ ] `prompt_tensor`
  - [ ] `target_reward`
  - [ ] `prompt_id / sample_id`

### 5.2 mid-step ranker（后做）
- [ ] 对中间 step 表示执行 MC 补全（例如 mc=4）。
- [ ] teacher target 使用 MC 后 reward 均值。
- [ ] 验证 scheduler 状态隔离（避免 step index 污染）。

---

## 6. Ranker 训练策略

- [ ] v1（推荐起步）：pointwise MSE。
- [ ] v2：`MSE + listwise KL`（提升排序鲁棒性）。
- [ ] 训练配置固定并记录：
  - [ ] batch size
  - [ ] lr / wd / epochs
  - [ ] warmup / scheduler
  - [ ] early stop 指标（Spearman 或 top-1 agreement）
- [ ] 验证指标：
  - [ ] Pearson / Spearman（ranker vs teacher）
  - [ ] Top-1 agreement（同一 prompt 内）
  - [ ] NDCG@k（可选）

---

## 7. Baseline 设计与对齐

统一预算原则：比较时尽量对齐 reward evaluation 次数或总候选数。

- [ ] `Base`：单次采样，不 rerank。
- [ ] `Best-of-N + CLIPScore`
- [ ] `Best-of-N + ImageReward`
- [ ] `Best-of-N + Ranker`
- [ ] `FK + CLIPScore`（difference / max）
- [ ] `FK + ImageReward`（difference / max）
- [ ] `FK + Ranker`（difference / max）

每个 baseline 需要：
- [ ] 统一输入池（同一组候选）
- [ ] 统一输出协议（selected image、selected id、selected score）
- [ ] 统一日志（runtime、GPU mem、fail count）

---

## 8. 评测与统计

### 8.1 自动指标
- [ ] GenEval 子任务分数 + overall。
- [ ] CLIPScore（作为参考，不作为唯一目标）。
- [ ] ImageReward / PickScore/HPSv2（偏好维度）。
- [ ] FID/IS（若有真实分布参考；否则说明其适用边界）。

### 8.2 统计显著性
- [ ] 至少 3 seeds 重复。
- [ ] 报告 mean/std。
- [ ] 关键比较做 paired test（prompt-level）。

### 8.3 可视化
- [ ] `N` 扩展曲线（N=1,2,4,8,10,16）。
- [ ] 预算-性能曲线（compute vs score）。
- [ ] 不同 potential（difference vs max）对比图。

---

## 9. 消融实验（建议最小集合）

- [ ] ranker输入层位消融（浅层/中层/深层/融合）。
- [ ] step_ratio 消融（early/mid/late）。
- [ ] teacher 选择消融（IR vs Pick/HPS）。
- [ ] FK 温度 `tau` 消融。
- [ ] particle size 消融（5/10/20）。

---

## 10. 工程稳定性与性能

- [ ] 所有 reward 模型只初始化一次（进程级缓存）。
- [ ] dataloader / scorer 全 batched。
- [ ] 大 tensor 保存前后做 shape 与 dtype 校验。
- [ ] offline 模式开关统一管理（HF/transformers）。
- [ ] 加 watchdog 日志（每 X 秒打印进度与吞吐）。

---

## 11. 论文结果对接（可直接映射到章节）

- [ ] 主表：各 baseline 在 GenEval/DrawBench/HPS 上表现。
- [ ] 主图：inference-time scaling 曲线（best-of-N vs FK vs ranker）。
- [ ] 消融表：layer/step/teacher/tau。
- [ ] 案例图：成功和失败样例（含 prompt）。
- [ ] 附录：运行配置与复现命令。

---

## 12. 里程碑排期（建议）

### Milestone A（2-3 天）
- [ ] 跑通候选池生成 + CLIP/IR best-of-N + GenEval主评测。

### Milestone B（2-3 天）
- [ ] 跑通 FK（CLIP/IR，difference/max）并完成预算对齐比较。

### Milestone C（3-5 天）
- [ ] 训练并接入 ranker（best-of-N + FK），完成主表和曲线。

### Milestone D（1-2 天）
- [ ] 做最小消融 + 清理脚本 + 固化复现文档。

---

## 13. 最小可运行命令清单（占位）

> 以下命令是占位模板，等脚本文件落地后再替换为真实命令。

- [ ] `bash text2image/scripts/run_all_baselines.sh`
- [ ] `bash text2image/scripts/run_all_fk.sh`
- [ ] `bash text2image/scripts/run_all_ranker.sh`
- [ ] `python text2image/eval/aggregate_metrics.py --input outputs/... --out outputs/summary.csv`
- [ ] `python text2image/eval/plot_curves.py --csv outputs/summary.csv --out outputs/figs/`

---

## 14. 完成定义（Definition of Done）

- [ ] 你可以一条命令重跑全部主实验。
- [ ] 任一方法失败可定位（日志中有 prompt_id、seed、阶段信息）。
- [ ] 主表、主图、消融表均可自动重生成。
- [ ] 代码与论文叙述一致（奖励定义、预算定义、对比设置一致）。

