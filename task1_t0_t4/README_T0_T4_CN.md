# Task 1：T0–T4 受控性能分解实验

本实验用于回答一个尚未解决的问题：

> 为什么逐链运行时，ThermoBayes 的轻量 Torch SGNHT 比 POSTERIORS SGNHT 快？

它不是再次比较预测精度，而是把同一条 POSTERIORS-order SGNHT 更新管线逐层“加回去”，定位额外时间出现在哪一层。

## 五个核心版本

| 版本 | 梯度接口 | 状态表示 | 更新执行 |
|---|---|---|---|
| T0 | `torch.autograd.grad` | flat tensor | 手写、out-of-place |
| T1 | `torch.func.grad_and_value` | flat tensor | 手写、out-of-place |
| T2 | `grad_and_value` | `SGNHTState` + tree utilities | 手写、out-of-place |
| T3 | POSTERIORS 内部接口 | `SGNHTState` + tree utilities | 原生 `transform.update(inplace=False)` |
| T4 | POSTERIORS 内部接口 | `SGNHTState` + tree utilities | 原生 `transform.update(inplace=True)` |

其中 **T3 是 POSTERIORS 默认行为，也是需要解释的主要终点**。T4 是优化控制，用来测量 out-of-place 状态创建的成本。

相邻差值的含义：

- `T1 - T0`：`torch.func.grad_and_value` 与 flat gradient wrapper 的影响。
- `T2 - T1`：`SGNHTState`、`tree_ravel`、`flexi_tree_map` 和新状态构造的影响。
- `T3 - T2`：POSTERIORS 原生 transform 调度、错误保护和 schedule 检查等剩余影响。
- `T3 - T4`：启用 `inplace=True` 能节省的时间。

前三项严格相加等于受控实验中的 `T3 - T0`。

## 公平性控制

所有版本共用：

- 相同初始参数、动量和 thermostat；
- 相同 minibatch index plan；
- 相同噪声种子；
- 相同四链逐链循环；
- 相同模型、数据和超参数；
- 相同 POSTERIORS 更新顺序。

脚本还会运行 trajectory parity test。只有 T0–T4 的最终 `params/momenta/xi` 都与 T3 近似一致，运行时分解才可解释。

## 安装依赖

```bash
python -m pip install posteriors
```

## 推荐命令

先进行短烟雾测试：

```bash
python scripts/run_t0_t4_ablation.py \
  --dataset yacht --id 1 --device cpu \
  --warmup_steps 10 --steps 50 --repeats 2 \
  --output_dir result_task1_t0_t4_smoke
```

RTX 5080 上的正式测试：

```bash
python scripts/run_t0_t4_ablation.py \
  --dataset yacht --id 1 --device cuda \
  --num_chains 4 --batch_size 32 \
  --warmup_steps 200 --steps 5000 --repeats 7 \
  --step_size 0.001 --diffusion_factor 0.01 \
  --seed 77210 --reference_ratio 2.22 \
  --output_dir result_task1_t0_t4_fold1
```

建议 Fold 1、4、7 分别执行一次。这里的 `steps` 表示四条链各前进一步的 round 数，因此 `5000 × 4 = 20000` 次梯度调用/版本/重复。

## 输出文件

- `t0_t4_repeats.csv`：每次重复的原始运行时间。
- `t0_t4_summary.csv`：每个版本的 median/mean/std、calls/s 和峰值 GPU 内存。
- `t0_t4_decomposition.json`：T0–T3 的差距分解及 T4 inplace 节省量。
- `t0_t4_parity.json`：轨迹一致性检查。
- `t0_t4_environment.json`：Python、PyTorch、POSTERIORS、GPU 等环境。
- `t0_t4_report.md`：自动生成的解释报告。

## 何时能够得出结论

先看 `T3/T0` 是否接近完整采样实验中观察到的 POSTERIORS/plain 比值。

- 若接近，例如误差在 15% 内：可以把 T0–T3 中最大的增量视为主要原因。
- 若明显不接近：T0–T4 只解释了 update kernel，剩余原因位于 kernel 外，例如 batch 生成、外层 Python 循环、初始化、样本收集或评价流程。此时需要 profiler，而不能直接停止调查。

不要因为某个单次重复很慢就下结论。优先使用 median，并检查不同执行顺序下结果是否稳定。
