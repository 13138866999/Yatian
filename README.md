# Critical Reproduction: Distribution-Aware Reweighting (HAM10000)
# 批判性复现：基于分布感知的皮肤病变分类重加权

> "Theory is continuous, but data is discrete - and scarce."
> “理论是连续的，但数据是离散的——且极其稀缺。”

---

## 1. 项目概述 (Project Overview)

### 🇨🇳 中文说明
本项目复现了论文 "Mitigating Individual Skin Tone Bias in Skin Lesion Classification through Distribution-Aware Reweighting" 的核心方法。

原论文提出了一种范式转变：从基于粗糙的离散类别（如 Fitzpatrick Skin Types I-VI）的公平性评估，转向基于 ITA (Individual Typology Angle) 的连续分布评估。通过 核密度估计 (KDE) 对肤色分布进行建模，并根据样本在分布空间中的“距离”进行反向加权 (Inverse Reweighting)，理论上可以消除对少数派肤色的偏见。

已实现的加权度量 (Implemented Metrics):
* FS (Fidelity Similarity / Bhattacharyya): 衡量分布重叠度。
* WD (Wasserstein Distance 1-D): 衡量将一个分布变换为另一个分布所需的“功”（推土机距离）。
* PF (Patrick-Fisher Distance): 分布间的欧氏距离。
* BS (Baseline): 基准线，无加权 (权重恒为 1.0) 以供对比。

### English Description
This project reproduces the methodology from "Mitigating Individual Skin Tone Bias in Skin Lesion Classification through Distribution-Aware Reweighting".

The paper proposes shifting fairness evaluation from discrete subgroups (Fitzpatrick Skin Types) to a continuous distribution based on Individual Typology Angle (ITA). By using Kernel Density Estimation (KDE), the method reweights the loss function to mitigate bias against underrepresented skin tones.

Implemented Metrics:
* FS (Fidelity Similarity / Bhattacharyya): Measures distribution overlap.
* WD (Wasserstein Distance 1-D): Measures the work to transform one distribution to another.
* PF (Patrick-Fisher Distance): Euclidean distance between distributions.
* BS (Baseline): Unweighted loss (weight = 1.0) for comparison.

---

## 2. 复现心得与数据挑战 (Reproduction Insights & Data Challenges)

### 🇨🇳 观察与思考
在复现过程中，我成功实现了论文提出的数学逻辑，但在 HAM10000 数据集上进行实验时，我观察到了一些值得深思的现象。这让我意识到理论算法在特定数据环境下落地的局限性。

1.  理论与数据的落差 (The Gap between Theory and Data)：
    HAM10000 数据集存在极度的不平衡。深色皮肤样本（Type 5 & 6）非常稀缺。在标准的测试集划分下，深色皮肤样本可能不足 10 个。这使得复杂的分布加权算法难以发挥全部潜力，因为“长尾”部分的样本几乎不存在。

2.  小样本带来的评估偏差 (Evaluation Bias from Small Samples)：
    实验数据显示 Type 5 和 6 的准确率经常接近 100%。经过分析，这并非模型在该群体上表现完美，而是样本量过小导致的统计波动。当测试样本只有个位数时，模型只需蒙对几张图，指标就会虚高，这掩盖了真实的泛化能力。

3.  学习总结 (Learning Outcome)：
    目前的实验结果表明，在如此小规模的数据集上，复杂的分布加权方法（FS/WD）与不加权的基准（BS）相比，没有表现出显著的性能差异。这教会我一个重要的道理：算法的公平性优化高度依赖于数据的代表性。如果没有足够多样化的数据支持，再先进的数学模型也难以从根本上解决偏见问题。

### 🇺🇸 Observations & Reflections
During the reproduction, I successfully implemented the mathematical logic proposed in the paper. However, experiments on the HAM10000 dataset revealed insightful challenges regarding the application of theory to real-world data.

1.  The Gap between Theory and Data:
    HAM10000 is severely imbalanced, with dark skin tones (Type 5 & 6) being negligible. In a standard test split, there may be fewer than 10 samples of dark skin. This scarcity limits the potential of distribution-aware reweighting, as the "tail" of the distribution is virtually missing.

2.  Evaluation Bias from Small Samples:
    I observed that accuracy for Type 5 & 6 often approaches 100%. Upon analysis, this is likely an artifact of small sample size rather than model superiority. With single-digit sample counts, correct predictions on just a few images can disproportionately inflate metrics, masking true generalization performance.

3.  Learning Outcome:
    Empirical results show that complex distribution reweighting (FS/WD) offers no significant advantage over the unweighted baseline (BS) in this specific setting. This highlights a key lesson: Algorithmic fairness is intrinsically tied to data representation. Without a sufficiently diverse dataset, advanced mathematical models struggle to mitigate bias effectively.
    
---
## 3. 使用说明

### 安装依赖

```bash
pip install -r requirements.txt
```

如果你没有虚拟环境，建议创建：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 训练与测试

主入口：

```bash
python /root/skinai/code/overall_q.py
```

常用参数：

```bash
python /root/skinai/code/overall_q.py \
  --csv-path /root/skinai/data/GroundTruth.csv \
  --output-dir /root/skinai/data/qi \
  --mode fs \
  --epochs 10 \
  --batch-size 256 \
  --num-folds 7 \
  --learning-rate 1e-5 \
  --seed 42 \
  --seed 42
```

仅测试：

```bash
python /root/skinai/code/overall_q.py --run-test-only
```

预处理配置（可选）：

```bash
python /root/skinai/code/overall_q.py --preprocess-config /path/to/preprocess.json
```

配置示例：

```json
{
  "steps": ["load_raw", "diagnosis", "merge_ita", "merge_meta", "clean", "validate"],
  "drop_duplicates": true,
  "drop_missing": true,
  "require_meta": true,
  "merge_how": "inner"
}
```

### 质量特征预处理

单独生成 ITA 质量特征：

```bash
python /root/skinai/code/preprocessing.py
```

### 结果对比与图表

汇总 bs/fs/wd/pf 的测试肤色分组均值并生成图表：

```bash
python /root/skinai/code/compare_skin_tone.py
```

输出路径：

- /root/skinai/data/results/comparison/testing_avg_per_skin_tone_all_modes.csv
- /root/skinai/data/results/comparison/comparison_f1_macro_avg.png
- /root/skinai/data/results/comparison/comparison_accuracy_avg.png
