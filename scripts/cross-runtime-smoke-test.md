# Cross-runtime wheel smoke test

`scripts/cross-runtime-smoke-test.sh` is a manual helper for checking that a locally
built LEAP GPU wheel can be loaded after swapping to a different accelerator runtime
module without rebuilding.

The script is intentionally not CI infrastructure. It records the installed
`libleapct.so` dynamic dependencies with `readelf`, swaps modules, verifies the
post-swap dynamic loader view with `ldd`, and runs a configurable Python smoke
command. The default smoke command only imports `leapctype`, constructs a
`tomographicModels` instance, and calls `about()`; it does not run a reconstruction.
Pass `--smoke-command` when a fuller site-specific reconstruction or numerical
comparison is desired.

## ROCm examples

Build against ROCm 6.3.1 and then load ROCm 6.4.3:

```bash
scripts/cross-runtime-smoke-test.sh \
    --build-module rocm/6.3.1hangfix \
    --run-module rocm/6.4.3leakfix \
    --gpu AMD \
    --variant rocm6.3 \
    --keep-work-dir
```

Build against ROCm 6.4.3 and then load ROCm 7.2.4:

```bash
scripts/cross-runtime-smoke-test.sh \
    --build-module rocm/6.4.3leakfix \
    --run-module rocm/7.2.4leakfix \
    --gpu AMD \
    --variant rocm6.4 \
    --keep-work-dir
```

Test the opposite direction where practical:

```bash
scripts/cross-runtime-smoke-test.sh \
    --build-module rocm/7.2.4leakfix \
    --run-module rocm/6.4.3leakfix \
    --gpu AMD \
    --variant rocm7.2 \
    --keep-work-dir
```

## CUDA example

```bash
scripts/cross-runtime-smoke-test.sh \
    --build-module cuda/12.0 \
    --run-module cuda/12.4 \
    --gpu NVIDIA \
    --variant cu120 \
    --keep-work-dir
```

## Environment assumptions

The helper assumes offline builds if not provided a venv. It sets
`PIP_NO_INDEX=1` and uses `.wheelhouse` by default. Use `--venv` or
`--wheelhouse` to override those defaults.

For ROCm module checks, the script expects module loading to set `ROCM_PATH`.
After swapping to the run module, it warns if `ldd` output does not include the
new `ROCM_PATH`, because that usually means the module swap did not affect the
libraries that the installed extension resolves at runtime.
