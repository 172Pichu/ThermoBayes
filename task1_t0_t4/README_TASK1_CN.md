# 任务 1：SGNHT 与 POSTERIORS 性能差距消融

本版本保留原始 ThermoBayes SGNHT，不修改其论文型 epsilon 离散化。新增方法仅用于对比实验。

## 新增方法

| CLI 方法 | 更新顺序 | 多链执行 | 用途 |
|---|---|---|---|
| `sgnht_posteriors_order` | POSTERIORS 顺序 | 每链独立 | 只改变更新顺序，与原始 `sgnht` 比较 |
| `sgnht_posteriors_order_vectorized` | POSTERIORS 顺序 | batched | 测量多链向量化收益 |
| `sgnht_posteriors_order_vectorized_compile` | POSTERIORS 顺序 | batched + step compile | 与现有 compile 路线对称 |
| `posteriors_sgnht_inplace` | POSTERIORS 原生 | 每链独立 | 测量 `inplace=True` 对状态分配开销的影响 |

原有方法 `sgnht`、`sgnht_vectorized`、`sgnht_vectorized_compile`、`posteriors_sgnht` 均保留。

## 更新顺序

原始 ThermoBayes：

1. 更新动量 `p`
2. 使用新动量更新参数 `x`
3. 使用新动量更新 thermostat `xi`

POSTERIORS-order：

1. 使用旧动量更新参数 `x`
2. 使用旧动量更新 thermostat `xi`
3. 使用旧 thermostat 更新动量 `p`

除了上述顺序，新增 ThermoBayes 对照版本继续使用原有数据、模型、随机预算、超参数、评价和保存管线。

## 推荐的第一轮实验

第一轮不要调参，固定原有配置。Windows PowerShell 或 Anaconda Prompt 中，在项目根目录执行：

```bash
python scripts/run_task1_comparison.py --id 1 --device cuda
```

如果当前环境没有安装 POSTERIORS，脚本会自动跳过两个原生 POSTERIORS 方法。安装后再次运行即可：

```bash
python -m pip install posteriors
python scripts/run_task1_comparison.py --id 1 --device cuda
```

加入两个 compile 版本：

```bash
python scripts/run_task1_comparison.py --id 1 --device cuda --include_compile
```

跑十折：

```bash
python scripts/run_task1_comparison.py --id -1 --device cuda
```

结果默认保存到 `result_task1/`，汇总表保存到：

```text
result_task1/_summaries/task1_sgnht_ablation_yacht_runs.csv
result_task1/_summaries/task1_sgnht_ablation_yacht_mean_std.csv
result_task1/_summaries/task1_sgnht_ablation_yacht_mean_std.md
```

## 关键比较

1. `sgnht` vs `sgnht_posteriors_order`：更新顺序影响。
2. `sgnht_posteriors_order` vs `sgnht_posteriors_order_vectorized`：多链向量化影响。
3. `sgnht_posteriors_order` vs `posteriors_sgnht`：相同顺序下 ThermoBayes 扁平张量管线与 POSTERIORS 框架的差异。
4. `posteriors_sgnht` vs `posteriors_sgnht_inplace`：POSTERIORS 状态复制/分配影响。

每个采样结果的 metadata 现在记录：

- `update_order`
- `vectorized_chains`
- `grad_eval_calls`
- `posteriors_inplace`（仅原生 POSTERIORS）
- `compiled_step`（compile 版本）

## 测试

```bash
python -m pytest -q
```

其中与 POSTERIORS 原生一步更新的一致性测试是可选测试：未安装 `posteriors` 时会自动跳过，其余测试仍会运行。
