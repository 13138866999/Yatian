## 使用说明

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
