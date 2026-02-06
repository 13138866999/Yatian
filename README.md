# Critical Reproduction: Distribution-Aware Reweighting for Skin Lesion Classification

> "Theory is continuous, but data is discrete - and scarce."

## 1. Project Overview

[cite_start]This project reproduces the methodology from "Mitigating Individual Skin Tone Bias in Skin Lesion Classification through Distribution-Aware Reweighting" (arXiv:2512.08733)[cite: 1].

The paper proposes shifting fairness evaluation from discrete subgroups (Fitzpatrick Skin Types) to a continuous distribution based on Individual Typology Angle (ITA). By using Kernel Density Estimation (KDE), the method reweights the loss function to mitigate bias against underrepresented skin tones.

### Implemented Metrics
[cite_start]We implemented the following distance-based metrics for loss reweighting[cite: 8]:
* FS (Fidelity Similarity / Bhattacharyya): Measures distribution overlap.
* WD (Wasserstein Distance 1-D): Measures the work to transform one distribution to another.
* PF (Patrick-Fisher Distance): Euclidean distance between distributions.
* BS (Baseline): Unweighted loss (weight = 1.0) for comparison.

## 2. Experimental Results & Critical Analysis

The experimental results on the HAM10000 dataset indicate a fundamental disconnect between the theoretical proposal and practical utility on small, imbalanced datasets.

### Empirical Observation
As shown in our experimental comparison (see `results/` for detailed logs):

1.  Negligible Improvement over Baseline:
    The Distance-Based Reweighting (DRW) methods (FS, WD, PF) fail to demonstrate a consistent or significant advantage over the unweighted Baseline (BS). In majority groups (Type 1-3), the Baseline often performs equally well or better.

2.  The Saturation Artifact (Type 5 & 6):
    Performance metrics for Type 5 and 6 skin tones approach 1.0 across all methods. This is not evidence of model superiority but an artifact of extreme sample scarcity. When the test set contains fewer than 10 samples (common in HAM10000 splits), a single correct classification disproportionately inflates the score.

3.  The Type 4 Drop:
    All methods, including the proposed reweighting schemes, suffer a significant performance drop in Type 4 skin tones. This suggests that the bottleneck is likely feature ambiguity or dataset noise, which cannot be solved simply by adjusting loss weights.

### Conclusion
While the distribution-aware framework is mathematically sound for continuous attributes, its application to HAM10000 yields results that are statistically indistinguishable from the baseline.

The "improvements" often cited in similar literature likely stem from the high variance inherent in small-sample testing rather than true algorithmic gain. For this method to be validated effectively, it requires a dataset with a statistically significant population of dark skin tones (e.g., >100 samples per type in the test set).

## 3. Usage

### Preprocessing
Generate ITA histograms and median values:
```bash
python overall_q.py --mode fs --epochs 0
