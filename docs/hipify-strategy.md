# LEAP's hipify strategy: vendoring, updates, and a future path to hipify-clang

This document explains why and how LEAP vendors CUDA→HIP translation logic for its AMD
build, how to update the vendored copy, and what would be involved in switching to ROCm's
official `hipify-clang` later, should that become worthwhile.

## What LEAP uses, and why

LEAP's AMD/HIP build needs to translate CUDA source (texture objects, cuFFT calls, etc.)
into HIP-compatible C++ before handing it to `hipcc`/`amdclang++`. This translation step —
"hipify" — is a separate concern from compiling; LEAP's own C++ has no other dependency on
either PyTorch or the HIPIFY toolchain.

LEAP vendors **[ROCm/hipify_torch](https://github.com/ROCm/hipify_torch)**, chosen over two
alternatives:

- **PyTorch's own internal copy** (`torch.utils.hipify`) — same lineage, but its API is
  mid-transition to a "v2" calling convention (a live deprecation warning as of this
  writing), and its CUDA→HIP mapping table has been trimmed toward what PyTorch's own
  kernels touch (3,517 lines, vs. `hipify_torch`'s 8,632) — narrower coverage for a general
  CUDA/cuFFT/texture-object codebase like LEAP's.
- **ROCm's official `hipify-clang`/`hipify-perl`** (from
  [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY)) — a real alternative, considered
  seriously, and the likely long-term destination. See "Future path" below for why it
  wasn't chosen for this pass.

`hipify_torch` is a pure-stdlib Python tool (`argparse`, `re`, `json`, `enum` — no
third-party pip dependencies, no torch, no CUDA toolkit needed to run it) that does
text/regex-based substitution against a large, hand-maintained CUDA→HIP symbol mapping
table. It ships a ready-made CMake API (`hipify()`, `get_hipified_list()` in
`cmake/Hipify.cmake`) explicitly intended for exactly this use case.

## Licensing and provenance

`hipify_torch` is MIT-licensed, copyright Advanced Micro Devices, Inc. This is the same
tool AMD originally wrote (2015-2016) and later contributed into PyTorch's own build
(jointly copyrighted AMD + Facebook from 2017-2018) before re-extracting it as this
standalone project. LEAP is MIT-licensed too, so vendoring is licensing-clean; carry the
copyright notice forward (this document plus the vendored `LICENSE.txt` satisfy that).

## Vendoring mechanism: `git subtree`, whole repo, squashed

The full upstream repo is small (704 KB, 16 files, 99 commits) — squashing removes the
commit-history weight, leaving a single ~700 KB addition to LEAP's tree. The repo is small
enough that trimming it to only the load-bearing files isn't worth the added complexity;
the whole repo is vendored as-is, at `third_party/hipify_torch/`.

**Initial vendoring (one-time):**

```bash
git remote add hipify_torch_upstream https://github.com/ROCm/hipify_torch.git
git fetch hipify_torch_upstream

# Pin to a specific, tested commit -- upstream has no tags/releases.
git subtree add --prefix=third_party/hipify_torch hipify_torch_upstream <PINNED_SHA> --squash
```

This produces one squashed commit in LEAP's history, with `git-subtree-dir` and
`git-subtree-split` trailers recording the upstream path and commit SHA — a real,
tool-recognized provenance record, in addition to this document.

**Updating later:**

```bash
git fetch hipify_torch_upstream
git subtree pull --prefix=third_party/hipify_torch hipify_torch_upstream <NEW_SHA> --squash
```

This is a real `git merge` under the hood (with the subtree strategy), not a blind
overwrite — if the local patch (below) and an upstream change touch the same lines, git
will report a normal merge conflict for manual resolution, rather than silently discarding
either side.

**After every update, run the checklist:**

1. `grep -n "detect_hipify_v2" third_party/hipify_torch/hipify_cli.py` — confirm the local
   patch (below) is still present; resolve any merge conflict here first if one occurred.
2. Re-run the CPU and AMD build validation against the updated copy before merging.
3. Update the pinned-SHA note in this document and in the subtree commit message.

## The local patch, and why it exists

`hipify_cli.py` (the script LEAP's CMake glue invokes) silently checks whether an installed
`torch` reports hipify version ≥2.0 and switches its own translation behavior accordingly —
with no explicit opt-out flag. That means the same LEAP source could hipify differently on
two machines depending on what happens to be `pip install`-ed, which undermines exactly the
build reproducibility this whole effort is after.

**Patch:** in `third_party/hipify_torch/hipify_cli.py`, `detect_hipify_v2()` is changed to
unconditionally `return False`, removing the `torch.utils.hipify` import probe entirely.
LEAP pins to v1 behavior deliberately; if v2 is ever wanted, that becomes an explicit,
reviewed one-line change here, not an environment-dependent accident.

```python
def detect_hipify_v2():
    # LEAP LOCAL PATCH: pin to v1 behavior deliberately, rather than silently
    # switching based on whatever torch happens to be installed in the build
    # environment. See docs/hipify-strategy.md. To opt into v2 behavior,
    # change this deliberately (and re-verify translation output).
    return False
```

Commit this as its own, clearly-labeled commit immediately after the `subtree add`, so it's
easy to find (`git log -- third_party/hipify_torch/hipify_cli.py`) and easy to re-apply or
reconcile after future `subtree pull`s.

## Known limitations of this approach

- The CUDA→HIP mapping table is a hand-maintained snapshot, versioned independently of both
  CUDA and ROCm. A newer CUDA 12.x-specific symbol LEAP starts using could simply be
  missing, surfacing as an unhelpful "not declared in this scope" compile error under
  `hipcc` rather than a clear translation warning.
- Regex/text substitution (vs. real AST parsing) can, in principle, mis-handle complex C++
  constructs — this hasn't caused a known problem yet, but hasn't been stress-tested against
  LEAP's texture-object and cuFFT-heavy code on real ROCm hardware either.
- This is genuinely a small, slow-moving side project relative to ROCm's main HIPIFY effort
  — worth periodically checking it hasn't gone stale.

## Future path: migrating to `hipify-clang`

### What would motivate this

- A real translation gap is hit: a CUDA 12.x (or later) API LEAP uses isn't in
  `hipify_torch`'s mapping table, and updating the vendored snapshot doesn't fix it because
  upstream hasn't added it either.
- LEAP needs to track a much newer or older ROCm release where the hand-maintained table
  has drifted noticeably from the real HIP API surface.
- A desire to drop vendored third-party code entirely (e.g., tightening supply-chain review
  requirements).
- Evidence that regex-based translation is mis-handling something complex in LEAP's actual
  kernels (texture objects, heavy template use) that AST-based translation would catch.

### What would be different

- **No vendoring at all.** `hipify-clang`/`hipify-perl` ship with every ROCm install;
  `find_program(HIPIFY_CLANG_EXECUTABLE hipify-clang HINTS ${ROCM_PATH}/bin)` replaces the
  entire `third_party/hipify_torch/` tree and its update process.
- **New CMake glue must be written from scratch** (~80-150 lines) — ROCm ships no
  reusable "hipify my sources" CMake function the way `hipify_torch` does. `hipify-clang`
  does accept multiple files in a single invocation, so this is likely one
  `execute_process()` call plus output-path bookkeeping, not a per-file loop. No Python
  needed at all — a real simplification over both the current approach and the earlier,
  abandoned 418-line Python wrapper.
- **CUDA toolkit headers become a build-time requirement on AMD-only nodes.**
  `hipify-clang` genuinely parses CUDA source via a real Clang AST and needs
  `--cuda-path` pointing at real CUDA headers to resolve includes — even on a machine with
  no NVIDIA GPU and no other reason to have CUDA installed. This is new provisioning burden
  that doesn't exist today.
- **Gains:** translation tracks whatever CUDA toolkit you point it at ("seamless support of
  new CUDA versions," per ROCm's own docs, since Clang does the parsing) and ships
  version-matched with whatever ROCm release is installed — directly relevant to LEAP's
  need to track specific ROCm/CUDA versions over time. AST-based parsing is also generally
  more robust against complex C++ than regex substitution.
- **Re-verify the historical friction first.** An earlier attempt at `hipify-clang` (see
  git history around the abandoned `packaging-updates` branch) hit real CLI errors on a
  ROCm 7.2.1 node — `-o` silently not writing output, `-p <dir>` conflicting with multiple
  source files, arch flags not propagating. That combination (compilation-database mode
  mixed with single-output mode) looks like a CLI misuse rather than a tool defect, but this
  needs a clean, minimal, direct invocation test on real hardware before committing to a
  migration, not just a read of the docs.
- **Licensing note:** `ROCm/HIPIFY` is Apache License 2.0 with LLVM Exceptions (verified
  directly, not assumed) — compatible with LEAP's MIT license, but Apache's NOTICE-file
  convention is a little more involved than MIT's; update `NOTICE`/license attribution
  accordingly if this migration happens.

### Suggested migration steps, when the time comes

1. On a real ROCm node (matching whatever ROCm version LEAP is targeting at the time), run
   a direct, minimal `hipify-clang` invocation against LEAP's actual `.cu`/`.cuh` files —
   not through any wrapper — and diff the output against what `hipify_torch` currently
   produces. Confirm it's clean before writing any CMake.
2. Write `cmake/LeapHipifyClang.cmake`: `find_program()` for the binary, wire up
   `--cuda-path` (from `find_package(CUDAToolkit)` or an explicit cache variable), pass
   LEAP's own include directories after the `--` separator, invoke once across all sources,
   and build the resulting file list for `target_sources()`.
3. Confirm CUDA toolkit headers are provisioned on whatever AMD-only build/CI nodes exist —
   this is new infrastructure, not just a code change.
4. Remove `third_party/hipify_torch/`, its CMake include, and the patch documented above.
5. Re-run the full sdist → wheel round-trip validation against the new path, on real ROCm
   hardware this time, before considering the migration done.
6. Update this document and `NOTICE` to reflect the new dependency and drop the
   `hipify_torch` attribution.
