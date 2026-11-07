#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <variant> <gpu>" >&2
    echo "Example: $0 rocm6.3 AMD" >&2
    echo "Example: $0 cu124 NVIDIA" >&2
    exit 2
fi

VARIANT="$1"
GPU="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WHEELHOUSE="${WHEELHOUSE:-${PROJECT_DIR}/.wheelhouse}"

export PIP_NO_INDEX="${PIP_NO_INDEX:-1}"
export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${WHEELHOUSE}}"

BASE_VERSION=$(python -c "from setuptools_scm import get_version; print(get_version(root='${PROJECT_DIR}'))")
CLEAN_BASE="${BASE_VERSION%%+*}"
FULL_VERSION="${CLEAN_BASE}+${VARIANT}"

cd "${PROJECT_DIR}"
SETUPTOOLS_SCM_PRETEND_VERSION="${FULL_VERSION}" \
    python -m build --wheel -Ccmake.define.LEAP_GPU="${GPU}" .
