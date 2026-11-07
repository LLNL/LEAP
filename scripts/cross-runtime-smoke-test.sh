#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/cross-runtime-smoke-test.sh \
      --build-module rocm/6.3.1hangfix \
      --run-module rocm/6.4.3leakfix \
      --gpu AMD \
      --variant rocm6.3

Build a wheel with one accelerator runtime module loaded, then switch to a
second runtime module without rebuilding and verify the installed shared library
resolves against the second runtime. By default this runs an import/about smoke
check, not a reconstruction. Pass --smoke-command '<python code>' to run a fuller
project-specific check.

Options:
  --build-module MODULE    Module used while building the wheel.
  --run-module MODULE      Module used while checking the installed wheel.
  --gpu AMD|NVIDIA|None    LEAP_GPU value passed to the wheel builder.
  --variant LABEL          Local version label, e.g. rocm6.3 or cu124.
  --venv PATH              Python environment to use for building.
                           Default: .venv under the project root.
  --wheelhouse PATH        Offline package source. Default: .wheelhouse.
  --smoke-command CODE     Python code run after the module swap.
  --work-dir PATH          Directory for temporary install and logs.
  --keep-work-dir          Do not delete the temporary work directory.
  -h, --help               Show this help.
EOF
}

BUILD_MODULE=""
RUN_MODULE=""
GPU=""
VARIANT=""
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_DIR}/.venv"
WHEELHOUSE="${PROJECT_DIR}/.wheelhouse"
SMOKE_COMMAND='import leapctype; model = leapctype.tomographicModels(); model.about()'
WORK_DIR=""
KEEP_WORK_DIR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-module)
            BUILD_MODULE="$2"
            shift 2
            ;;
        --run-module)
            RUN_MODULE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --venv)
            VENV="$2"
            shift 2
            ;;
        --wheelhouse)
            WHEELHOUSE="$2"
            shift 2
            ;;
        --smoke-command)
            SMOKE_COMMAND="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --keep-work-dir)
            KEEP_WORK_DIR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${BUILD_MODULE}" || -z "${RUN_MODULE}" || -z "${GPU}" || -z "${VARIANT}" ]]; then
    usage >&2
    exit 2
fi

if [[ -z "${WORK_DIR}" ]]; then
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/leap-cross-runtime.XXXXXX")"
else
    mkdir -p "${WORK_DIR}"
fi

cleanup() {
    if [[ "${KEEP_WORK_DIR}" -eq 0 ]]; then
        rm -rf "${WORK_DIR}"
    else
        printf 'Kept work directory: %s\n' "${WORK_DIR}"
    fi
}
trap cleanup EXIT

if ! type module >/dev/null 2>&1; then
    for init_script in /usr/share/lmod/lmod/init/bash /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "${init_script}" ]]; then
            # shellcheck disable=SC1090
            . "${init_script}"
            break
        fi
    done
fi

if ! type module >/dev/null 2>&1; then
    echo "The environment module command is not available." >&2
    exit 1
fi

export VIRTUAL_ENV="${VENV}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PIP_NO_INDEX=1
export PIP_FIND_LINKS="${WHEELHOUSE}"

python --version
printf 'Build module: %s\n' "${BUILD_MODULE}"
printf 'Run module:   %s\n' "${RUN_MODULE}"
printf 'Work dir:     %s\n' "${WORK_DIR}"

module load "${BUILD_MODULE}"
printf 'Build ROCM_PATH: %s\n' "${ROCM_PATH:-unset}"

"${PROJECT_DIR}/scripts/build-gpu-wheel.sh" "${VARIANT}" "${GPU}" 2>&1 | tee "${WORK_DIR}/build.log"
WHEEL=$(find "${PROJECT_DIR}/dist" -maxdepth 1 -type f -name "leapct-*+${VARIANT}-*.whl" -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 {print $2}')
if [[ -z "${WHEEL}" ]]; then
    echo "Could not find the built wheel for variant ${VARIANT}" >&2
    exit 1
fi
printf 'Wheel: %s\n' "${WHEEL}"

python -m venv "${WORK_DIR}/venv"
"${WORK_DIR}/venv/bin/python" -m pip install --no-index --find-links="${WHEELHOUSE}" "${WHEEL}"
LIB=$("${WORK_DIR}/venv/bin/python" - <<'PY'
from pathlib import Path
import leapctype
package_dir = Path(leapctype.__file__).resolve().parent
matches = sorted(package_dir.glob('*leapct*.so'))
if matches:
    print(matches[0])
else:
    print(package_dir / 'libleapct.so')
PY
)
if [[ ! -f "${LIB}" ]]; then
    echo "Installed LEAP shared library not found at ${LIB}" >&2
    exit 1
fi
printf 'Library: %s\n' "${LIB}"

readelf -d "${LIB}" | tee "${WORK_DIR}/needed-build.txt"

module unload "${BUILD_MODULE}" >/dev/null 2>&1 || true
module load "${RUN_MODULE}"
printf 'Run ROCM_PATH: %s\n' "${ROCM_PATH:-unset}"
ldd "${LIB}" | tee "${WORK_DIR}/ldd-run.txt"

if [[ "${GPU}" == "AMD" && -n "${ROCM_PATH:-}" ]]; then
    if ! grep -F "${ROCM_PATH}" "${WORK_DIR}/ldd-run.txt" >/dev/null; then
        echo "Warning: ldd output did not include ROCM_PATH=${ROCM_PATH}" >&2
    fi
fi

"${WORK_DIR}/venv/bin/python" -c "${SMOKE_COMMAND}"
printf 'Cross-runtime smoke check completed. Logs are in %s while this script is running.\n' "${WORK_DIR}"
