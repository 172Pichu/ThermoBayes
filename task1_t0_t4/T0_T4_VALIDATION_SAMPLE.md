# T0–T4 implementation validation sample

This file records a short implementation validation performed while packaging the experiment. It is not a substitute for rerunning on the user's research environment.

Environment:

- Dataset: Yacht Fold 1
- CPU, `torch_num_threads=1`
- PyTorch 2.10.0+cpu
- POSTERIORS 0.1.2
- 4 chains
- 20 warm-up rounds
- 200 measured rounds, 800 gradient calls per repeat
- 5 repetitions

Median result:

| Variant | Median seconds | Ratio to T0 |
|---|---:|---:|
| T0 flat + autograd | 0.267070 | 1.000× |
| T1 flat + grad_and_value | 0.477416 | 1.788× |
| T2 state/tree + manual update | 0.561166 | 2.101× |
| T3 native POSTERIORS out-of-place | 0.619639 | 2.320× |
| T4 native POSTERIORS in-place | 0.549267 | 2.057× |

The controlled `T3/T0 = 2.320×`, close to the historical full-run ratio of about `2.22×`. Numerical parity passed for all five variants.

For this validation only, the median `T3−T0` gap decomposed approximately as:

- T1−T0, gradient API and flat gradient wrapper: 59.7%
- T2−T1, state/tree and out-of-place state construction: 23.8%
- T3−T2, remaining native transform dispatch/guards: 16.6%
- T3−T4, time saved by `inplace=True`: about 11.4% of T3

These numbers are promising evidence that the experiment design reproduces the original phenomenon, but the final research conclusion must use results from the same machine and software environment as the main comparison.
