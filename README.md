# libuipc-samples

A sample library for Libuipc.

This library is a collection of sample programs that demonstrate how to use the Libuipc library.

## Install (uv recommended)

Install base dependencies (without `pyuipc`):

```bash
uv sync
```

### pyuipc from PyPI

```bash
uv sync --group pypi
```

> **Note:** The PyPI version of `pyuipc` only supports CUDA 12.8. If you need a different CUDA version, use the "Build from source" method below.

### pyuipc from source

Build `pyuipc` following the [Libuipc documentation](https://spirimirror.github.io/libuipc-doc/build_install/), then install the wheel:

```bash
uv pip install pyuipc-<version>.whl
```
