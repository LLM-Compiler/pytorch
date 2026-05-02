#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_advisor_roofline.sh
#
# Collects Intel Advisor cache-aware roofline data for two paths:
#   1. standard  — plain Inductor compilation (no NUMA passes)
#   2. numa_tp   — NUMA tensor-parallel Inductor compilation
#
# Output layout
# -------------
#   advisor_projects/
#     standard/          ← Advisor project dir for standard path
#     numa/              ← Advisor project dir for NUMA-TP path
#
# After this script finishes, run generate_advisor_reports.sh (or open the
# project dirs in the Intel Advisor GUI) to view/export the roofline charts.
#
# Prerequisites (all verified before running):
#   - Intel Advisor 2023.1.0 at /opt/intel/oneapi/advisor/2023.1.0
#   - conda env  triton_cpu_integrated  with the NUMA-enabled pytorch build
#   - perf_event_paranoid = 0  (already set on this system)
#   - /dev/cpu/*/msr  accessible  (msr kernel module loaded)
#   - uncore_imc PMU events available (cas_count_read/write confirmed)
#
# Two-pass strategy per mode
# --------------------------
# Pass 1 — survey:
#   Lightweight sampling pass.  Identifies hot loops and measures wall time.
#   Overhead: ~1-5×.
#
# Pass 2 — tripcounts + FLOP + cache simulation:
#   Counts FP/integer operations and models L1/L2/L3 cache traffic per loop.
#   Overhead: 30-100× (cache simulation is expensive — plan ~10-30 min per mode).
#   Result: Arithmetic Intensity (AI = FLOPs / Bytes) for every hot loop,
#   plotted against L1/L2/L3/DRAM bandwidth ceilings → cache-aware roofline.
#
# Both passes are launched with --start-paused.  The bench scripts call
# __itt_resume() / __itt_pause() around the measurement loop only, so the
# trace contains only steady-state Triton-CPU kernel execution (no
# torch.compile, no Triton-kernel JIT, no gcc, no autotuning).
#
# Environment variables you can override before running:
#   ADVISOR_BENCH_B          batch size            (default 1)
#   ADVISOR_BENCH_S          sequence length       (default 512)
#   ADVISOR_BENCH_TP_SHARDS  NUMA TP world size    (default 4, = num NUMA nodes)
#   ADVISOR_BENCH_WARMUP     warmup iters (paused) (default 2)
#   ADVISOR_BENCH_ITERS      measured iterations   (default 20)
#   ADVISOR_MODULE_FILTER    comma-separated module basenames to exclude
#                            (default skips gcc/ld/binutils/Python-crypto noise)
#
# Intel Advisor can appear to “hang” during survey at ~50% while processing
# “Dynamic Instruction mix” (it is analyzing many/large binaries). Workarounds:
#   - This script passes --static-instruction-mix on survey (faster; see Intel forums).
#   - Set ADVISOR_USE_DYNAMIC_INSTRUCTION_MIX=1 to omit that flag if you need dynamic mix.
#   - Set ADVISOR_MODULE_FILTER to a comma-separated list of basenames to exclude, e.g.
#       export ADVISOR_MODULE_FILTER='libfoo.so,libbar.so'
#     (adds --module-filter-mode=exclude --module-filter=... on survey + tripcounts).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# All paths below are env-overrideable — defaults match the standard Ubuntu setup
# (intel-oneapi-advisor apt package + Miniconda in $HOME/miniconda3).
ADVISOR_HOME="${ADVISOR_HOME:-/opt/intel/oneapi/advisor/latest}"
ADVISOR_BIN="${ADVISOR_HOME}/bin64/advixe-cl"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-triton_cpu_integrated}"
PYTHON="${PYTHON:-${CONDA_BASE}/envs/${CONDA_ENV}/bin/python}"

PROJECT_ROOT="${SCRIPT_DIR}/advisor_projects"
PROJ_STD="${PROJECT_ROOT}/standard"
PROJ_NUMA="${PROJECT_ROOT}/numa"

BENCH_STD="${SCRIPT_DIR}/advisor_bench_standard.py"
BENCH_NUMA="${SCRIPT_DIR}/advisor_bench_numa.py"

# ── sanity checks ─────────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="

[[ -x "${ADVISOR_BIN}" ]] || { echo "ERROR: advixe-cl not found at ${ADVISOR_BIN}"; exit 1; }
[[ -x "${PYTHON}" ]]      || { echo "ERROR: Python not found at ${PYTHON}"; exit 1; }

PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
echo "  perf_event_paranoid : ${PARANOID}  (need <= 0 for uncore PMU events)"
if [[ "${PARANOID}" -gt 0 ]]; then
    echo "  WARNING: perf_event_paranoid=${PARANOID}. Advisor may not measure DRAM bandwidth."
    echo "  To fix (as root): echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi

MSR_OK="no"
[[ -e /dev/cpu/0/msr ]] && MSR_OK="yes"
echo "  /dev/cpu/0/msr      : ${MSR_OK}  (needed for hardware PMU collection)"

echo "  Intel Advisor       : $("${ADVISOR_BIN}" --version 2>&1 | head -1)"
echo "  Python              : ${PYTHON}"
echo "  TP shards (NUMA)    : ${ADVISOR_BENCH_TP_SHARDS}"
echo "  Bench scripts       : ${BENCH_STD}"
echo "                        ${BENCH_NUMA}"
echo ""

# ── source Advisor environment ─────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${ADVISOR_HOME}/advisor-vars.sh" 2>/dev/null || true

# ── activate conda env  ────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${PROJ_STD}" "${PROJ_NUMA}"

# Survey: static instruction mix avoids lengthy dynamic binary analysis (Intel workaround).
ADVISOR_SURVEY_EXTRA=()
if [[ "${ADVISOR_USE_DYNAMIC_INSTRUCTION_MIX:-0}" != "1" ]]; then
    ADVISOR_SURVEY_EXTRA+=(--static-instruction-mix)
fi
ADVISOR_TRIP_EXTRA=()

# Default module filter: exclude gcc/ld/binutils/Python-crypto noise that gets
# pulled in only during torch.compile + Triton kernel compilation.  Triton runtime
# (libtriton.so), the Triton-CPU launcher (__triton_cpu_launcher.so), generated
# kernels (tmp*.so), libtorch_cpu.so, libgomp.so, OpenBLAS, and sleef are NOT in
# this list — they remain instrumented because they ARE the kernel-execution path.
DEFAULT_ADVISOR_MODULE_FILTER='cc1plus,x86_64-linux-gnu-as,x86_64-linux-gnu-ld.bfd,collect2,x86_64-linux-gnu-gcc-13,libcrypto.so.3,libssl.so.3,libgmp.so.10,libpindwarf.so,libbfd-2.42-system.so,libbfd-2.42.so,_ssl.cpython-39-x86_64-linux-gnu.so,_hashlib.cpython-39-x86_64-linux-gnu.so,_ctypes.cpython-39-x86_64-linux-gnu.so,_json.cpython-39-x86_64-linux-gnu.so,_elementtree.cpython-39-x86_64-linux-gnu.so'
ADVISOR_MODULE_FILTER="${ADVISOR_MODULE_FILTER:-${DEFAULT_ADVISOR_MODULE_FILTER}}"

if [[ -n "${ADVISOR_MODULE_FILTER}" ]]; then
    ADVISOR_SURVEY_EXTRA+=(--module-filter-mode=exclude --module-filter="${ADVISOR_MODULE_FILTER}")
    ADVISOR_TRIP_EXTRA+=(--module-filter-mode=exclude --module-filter="${ADVISOR_MODULE_FILTER}")
fi

# --start-paused: Advisor begins paused.  The bench scripts call __itt_resume() /
# __itt_pause() around the measurement loop ONLY, so the recorded trace excludes
# torch.compile, Triton kernel JIT, gcc invocations, and warmup iterations.
ADVISOR_PAUSE_FLAGS=(--start-paused)

# Pass through measurement controls so the bench's "warmup outside, measure inside"
# split is honored.
export ADVISOR_BENCH_WARMUP="${ADVISOR_BENCH_WARMUP:-2}"
export ADVISOR_BENCH_ITERS="${ADVISOR_BENCH_ITERS:-20}"
# NUMA TP world size — default to the host's NUMA node count if numactl is available,
# otherwise fall back to 2.  Override with ADVISOR_BENCH_TP_SHARDS=<n>.
if [[ -z "${ADVISOR_BENCH_TP_SHARDS:-}" ]]; then
    if command -v numactl >/dev/null 2>&1; then
        _detected_nodes=$(numactl --hardware 2>/dev/null | awk '/^available:/ {print $2}')
        ADVISOR_BENCH_TP_SHARDS="${_detected_nodes:-2}"
    else
        ADVISOR_BENCH_TP_SHARDS=2
    fi
fi
export ADVISOR_BENCH_TP_SHARDS
export ADVISOR_HOME

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one two-pass Advisor collection
# Usage: run_advisor_collection <project_dir> <bench_script> <label>
# ─────────────────────────────────────────────────────────────────────────────
run_advisor_collection() {
    local proj="$1"
    local bench="$2"
    local label="$3"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  MODE: ${label}"
    echo "════════════════════════════════════════════════════════════"

    # ── Pass 1: Survey ────────────────────────────────────────────────────────
    echo ""
    echo "── Pass 1 / 2: Survey (hot-loop sampling) ──"
    echo "   Overhead: ~1-5×.  Expected duration: a few minutes."
    if [[ "${#ADVISOR_SURVEY_EXTRA[@]}" -gt 0 ]]; then
        echo "   Extra survey flags: ${ADVISOR_SURVEY_EXTRA[*]}"
    fi
    echo ""
    "${ADVISOR_BIN}" \
        --collect=survey \
        "${ADVISOR_PAUSE_FLAGS[@]}" \
        "${ADVISOR_SURVEY_EXTRA[@]}" \
        --project-dir="${proj}" \
        --search-dir src:rp="${SCRIPT_DIR}" \
        -- "${PYTHON}" "${bench}"

    # ── Pass 2: Trip Counts + FLOP + Cache Simulation ─────────────────────────
    echo ""
    echo "── Pass 2 / 2: Trip Counts + FLOP + Cache Simulation ──"
    echo "   Overhead: 30-100×.  Expected duration: 10-40 minutes."
    echo "   This is the expensive pass — cache simulation models L1/L2/L3 traffic."
    if [[ "${#ADVISOR_TRIP_EXTRA[@]}" -gt 0 ]]; then
        echo "   Extra tripcounts flags: ${ADVISOR_TRIP_EXTRA[*]}"
    fi
    echo ""
    "${ADVISOR_BIN}" \
        --collect=tripcounts \
        --enable-cache-simulation \
        --flop \
        "${ADVISOR_PAUSE_FLAGS[@]}" \
        "${ADVISOR_TRIP_EXTRA[@]}" \
        --project-dir="${proj}" \
        --search-dir src:rp="${SCRIPT_DIR}" \
        -- "${PYTHON}" "${bench}"

    echo ""
    echo "  [${label}] collection complete → ${proj}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run both modes
# ─────────────────────────────────────────────────────────────────────────────
run_advisor_collection "${PROJ_STD}"  "${BENCH_STD}"  "standard (no NUMA)"
run_advisor_collection "${PROJ_NUMA}" "${BENCH_NUMA}" "numa_tp  (NUMA tensor-parallel)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Collection finished for both modes."
echo "  Next step: run  generate_advisor_reports.sh"
echo "  Or open in GUI (X11/RDP/VNC required):"
echo "    ${ADVISOR_HOME}/bin64/advixe-gui ${PROJ_STD}"
echo "    ${ADVISOR_HOME}/bin64/advixe-gui ${PROJ_NUMA}"
echo "  (You can also export an HTML/CSV report on this headless box via"
echo "   advixe-cl --report=roofline --project-dir=<proj> --report-output=<file>)"
echo "════════════════════════════════════════════════════════════"
