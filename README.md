# Critical Reproduction: Distribution-Aware Reweighting (HAM10000)
# 批判性复现：基于分布感知的皮肤病变分类重加权

> **"Theory is continuous, but data is discrete - and often biased."**
> **“理论是连续的，但数据是离散的——且往往充满偏见。”**

---

## 1. Project Overview (项目概述)

### 🇨🇳 中文说明
本项目复现了论文 "Mitigating Individual Skin Tone Bias in Skin Lesion Classification through Distribution-Aware Reweighting" 的核心方法。

本项目旨在验证一个关键假设：**在极度不平衡的小样本医疗数据集上，数学上的分布对齐是否真能带来公平性收益？**

原论文提出从离散类别（Fitzpatrick I-VI）转向基于 ITA (Individual Typology Angle) 的连续分布评估。通过 **核密度估计 (KDE)** 对肤色分布建模，并根据样本在分布空间中的“距离”进行反向加权。

**已实现的加权度量 (Implemented Metrics):**
* **FS (Fidelity Similarity / Bhattacharyya):** 衡量分布重叠度。
* **WD (Wasserstein Distance 1-D):** 衡量分布变换所需的“推土机距离”。
* **PF (Patrick-Fisher Distance):** 分布间的欧氏距离。
* **BS (Baseline):** 基准线，无加权 (权重恒为 1.0) 以供对比。

### 🇺🇸 English Description
This repository reproduces the methodology from "Mitigating Individual Skin Tone Bias in Skin Lesion Classification through Distribution-Aware Reweighting".

The project serves as a critical examination of the hypothesis: **Can mathematical distribution alignment genuinely mitigate bias in severely imbalanced, small-scale medical datasets?**

Moving beyond discrete subgroups (Fitzpatrick Skin Types), this method utilizes **Kernel Density Estimation (KDE)** on the continuous Individual Typology Angle (ITA) to reweight the loss function based on distribution distance.

---

## 2. Dataset Preparation (数据准备)

**⚠️ CRITICAL:** The code's `_resolve_paths` logic expects a strict directory structure relative to the source code. You must combine data from two sources into the `data/` directory.（you can also change the input_dir my parameter）

**⚠️ 注意：** 代码中的路径解析逻辑依赖于相对路径。请务必将两个来源的数据合并至 `data/` 目录中。(可通过参数调整输入地址）

### Data Sources (数据源)
1.  **Images & Masks (图像与分割掩码):**
    * Source: [HAM1000 Segmentation and Classification (Kaggle)](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
    * *Action:* Extract images to `data/images` and masks to `data/masks`.
2.  **Original Metadata (原始元数据) CVS only ---- 可以只把metadata.cvs下载下来（can download metadata.cvs only）:**
    * Source: [Skin Cancer MNIST: HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
    * *Action:* Place `HAM10000_metadata.csv` in `data/`. This is required for lesion-level split to prevent data leakage.

### Directory Structure (目录结构)

```text
.
├── code/                 # Source code (overall_q.py, etc.)
├── data/                 # DATASET ROOT
│   ├── images/           # [Required] All .jpg images
│   ├── masks/            # [Required] Segmentation masks (*_segmentation.png)
│   ├── HAM10000_metadata.csv  # [Required] Original metadata for lesion_id
│   ├── GroundTruth.csv   # [Input] Your main label file
│   └── results/          # [Output] Training results
└── requirements.txt

```

---

## 3. Usage (使用说明)

### Installation (安装)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### 1. Preprocessing (数据预处理)

Before training, you must generate the ITA (skin tone) quality features.
训练前必须生成 ITA 肤色质量特征。

```bash
# This will generate 'ita_medians.csv' in your output directory
python code/overall_q.py \
  --csv-path data/GroundTruth.csv \
  --output-dir data/qi

```

### 2. Training (模型训练)

Run the main training pipeline. You can customize the **input CSV** and **output directory**.
运行主训练流程。你可以自定义**输入 CSV** 和 **输出目录**。

**Example (FS Mode):**

```bash
python code/overall_q.py \
  --csv-path data/GroundTruth.csv \
  --output-dir data/qi \
  --mode fs \
  --epochs 10 \
  --batch-size 256 \
  --num-folds 7 \
  --learning-rate 1e-5 \
  --seed 42

```

### 3. Testing Only (仅测试)

If you have trained models and want to evaluate them on a test set:
如果已有训练好的模型并希望进行测试评估：

```bash
python code/overall_q.py \
  --csv-path data/GroundTruth.csv \
  --output-dir data/qi \
  --mode fs \
  --run-test-only

```

### 4. Comparison (结果对比)

Generate comparison plots for all modes (BS/FS/WD/PF).
生成所有模式的对比图表。

```bash
python code/compare_skin_tone.py

```

---

## 4. Arguments (参数说明)

| Argument | Default | Description |
| --- | --- | --- |
| `--csv-path` | `None` | **[Custom Input]** Path to the input CSV file (containing 'image', 'diagnosis'). |
| `--output-dir` | `None` | **[Custom Output]** Directory to save results, checkpoints, and logs. |
| `--mode` | `fs` | Reweighting mode: `bs` (Baseline), `fs`, `wd`, `pf`. |
| `--epochs` | `10` | Number of training epochs per fold. |
| `--batch-size` | `256` | Batch size for dataloaders. |
| `--num-folds` | `7` | Number of folds for Cross-Validation. |
| `--run-test-only` | `False` | Skip training and run evaluation on existing checkpoints. |
| `--seed` | `42` | Random seed for reproducibility. |

---

## 5. Critical Analysis (批判性分析)

### 🇨🇳 实验结论：局限性中的公平性曙光

本项目的实验揭示了在极端不平衡数据下，算法修正与数据质量之间的博弈。

1. **长尾分布的挑战与适应 (The Challenge of Long-Tail):**
   HAM10000 数据集中，深色皮肤（Type 5 & 6）样本极度稀缺（测试集常不足 10 例）。虽然这导致核密度估计 (KDE) 的稳定性受到挑战，但实验表明，分布加权算法（FS/WD）并未完全失效。相反，它在有限的样本空间内依然尝试捕捉分布差异，并给出了数学上合理的权重修正。

2. **微弱但积极的公平性信号 (Subtle yet Positive Fairness Signals):**
   尽管小样本导致指标波动较大，但对比数据可见，**引入分布加权（FS/WD/PF）后，稀缺样本（Type 5 & 6）的 F1-Macro 指标普遍优于无加权的基准线（BS）。** 这证明了算法在“关注少数派”这一核心目标上是生效的——即便在数据极度匮乏的情况下，反向加权机制依然帮助模型更准确地识别了边缘群体。

3. **结论 (Conclusion):**
   算法并非万能药，但它是一道有效的防线。实验证明，**数据代表性虽是根本，但在数据存在结构性缺失时，分布感知重加权（DRW）仍能提供比默认训练更优的公平性保障**，尽管这种提升在极小样本下显得较为微弱。

### 🇺🇸 Empirical Analysis: Fairness Amidst Scarcity

This reproduction highlights the nuanced interaction between algorithmic correction and data quality in extremely imbalanced settings.

1. **Constraints of the Long-Tail:**
   The extreme scarcity of dark skin tones (Type 5 & 6) in HAM10000 poses a significant challenge to Kernel Density Estimation (KDE). However, the distribution-aware algorithms (FS/WD) did not completely fail. Instead, they functioned within the limits of the data, attempting to model the distribution shift and apply logical reweighting even with minimal support.

2. **Marginal but Consistent Fairness Gains:**
   While small sample sizes introduce statistical noise, a direct comparison reveals a crucial trend: **Distribution-aware methods (FS/WD/PF) consistently achieved higher F1-Macro scores on the rare Type 5 & 6 categories compared to the unweighted Baseline (BS).** This validates the algorithm's core premise: the inverse reweighting mechanism successfully forced the model to prioritize underrepresented samples, mitigating bias to the extent allowed by the data.

3. **Conclusion:**
   Algorithms cannot fully compensate for structural data deficits, but they serve as a necessary safeguard. The results demonstrate that while **data representation is primary, Distribution-Aware Reweighting (DRW) offers a tangible, albeit subtle, improvement in fairness over standard training**, acting as a critical correction mechanism when diverse data is unavailable.

```

```
