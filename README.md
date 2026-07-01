# pyg-lib

> **macOS / Apple Silicon (MPS) fork.** This branch (`macos-mps-scatter`) adds
> MPS backend support: native `scatter_sum/mul/mean`, a **fused single-pass
> Metal kernel** for `scatter_min/max` (value + arg in one 64-bit-atomic pass,
> arbitrary `dim`/rank, float32/float16/bfloat16), and CPU-assisted shims for
> point-cloud/spline ops. See the upstream project at
> [pyg-team/pyg-lib](https://github.com/pyg-team/pyg-lib).
>
> **Install on macOS (tested with `uv`):**
>
> ```bash
> git clone -b macos-mps-scatter https://github.com/zzccppp/pyg-lib.git
> git -C pyg-lib submodule update --init --recursive \
>     third_party/METIS third_party/parallel-hashmap
> uv venv --python 3.12
> uv pip install torch setuptools wheel ninja
> uv pip install --no-build-isolation ./pyg-lib   # --no-build-isolation is required
> ```
>
> `--recursive` is needed because METIS has a nested `GKlib` submodule.

<div align="center">

[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Slack][slack-image]][slack-url]
[![Contributing][contributing-image]][contributing-url]

</div>

## Installation

We provide pre-built Python wheels for all major OS/PyTorch/CUDA combinations from Python 3.10 till 3.14, see [here](https://data.pyg.org/whl).

To install the wheels, simply run

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

- `${TORCH}` should be replaced by either `2.10.0`, `2.11.0`, or `2.12.0`
- `${CUDA}` should be replaced by either `cpu`, `cu126`, `cu128`, `cu130`, or `cu132`

The following combinations are supported:

| PyTorch 2.12 | `cpu` | `cu126` | `cu128` | `cu130` | `cu132` |
| ------------ | ----- | ------- | ------- | ------- | ------- |
| **Linux**    | ✅    | ✅      |         | ✅      | ✅      |
| **Windows**  | ✅    | ✅      |         | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |         |

| PyTorch 2.11 | `cpu` | `cu126` | `cu128` | `cu130` |
| ------------ | ----- | ------- | ------- | ------- |
| **Linux**    | ✅    | ✅      | ✅      | ✅      |
| **Windows**  | ✅    | ✅      | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |

| PyTorch 2.10 | `cpu` | `cu126` | `cu128` | `cu130` |
| ------------ | ----- | ------- | ------- | ------- |
| **Linux**    | ✅    | ✅      | ✅      | ✅      |
| **Windows**  | ✅    | ✅      | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |

### From nightly

Nightly wheels are provided for Linux from Python 3.10 till 3.14:

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html
```

### From master

```
pip install ninja wheel
pip install --no-build-isolation git+https://github.com/pyg-team/pyg-lib.git
```

[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&color=4B26A4
[contributing-url]: https://github.com/pyg-team/pytorch_geometric/blob/master/.github/CONTRIBUTING.md
[coverage-image]: https://codecov.io/gh/pyg-team/pyg-lib/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/pyg-team/pyg-lib?branch=master
[docs-image]: https://readthedocs.org/projects/pyg-lib/badge/?version=latest
[docs-url]: https://pyg-lib.readthedocs.io/en/latest/?badge=latest
[slack-image]: https://img.shields.io/badge/slack-join-white.svg?logo=slack&color=4B26A4
[slack-url]: https://data.pyg.org/slack.html
