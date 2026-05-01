#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_advisor_reports.sh
#
# Exports roofline and loop-performance data from the two Advisor project
# directories created by run_advisor_roofline.sh.
#
# Outputs (under advisor_projects/reports/):
#   roofline_standard.html   — interactive cache-aware roofline chart (standard)
#   roofline_numa.html       — interactive cache-aware roofline chart (NUMA-TP)
#   topdown_standard.csv     — per-loop self-time + AI + GFLOPs/s (standard)
#   topdown_numa.csv         — per-loop self-time + AI + GFLOPs/s (NUMA-TP)
#   tripcounts_standard.csv  — per-loop trip counts + FLOP / byte traffic (standard)
#   tripcounts_numa.csv      — per-loop trip counts + FLOP / byte traffic (NUMA-TP)
#   survey_standard.txt      — human-readable survey report (standard)
#   survey_numa.txt          — human-readable survey report (NUMA-TP)
#   roofline_comparison.txt  — side-by-side summary printed to terminal
#
# Open the full interactive roofline charts in the Advisor GUI:
#   advixe-gui advisor_projects/standard
#   advixe-gui advisor_projects/numa
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVISOR_HOME="/opt/intel/oneapi/advisor/2023.1.0"
ADVISOR_BIN="${ADVISOR_HOME}/bin64/advixe-cl"

PROJECT_ROOT="${SCRIPT_DIR}/advisor_projects"
PROJ_STD="${PROJECT_ROOT}/standard"
PROJ_NUMA="${PROJECT_ROOT}/numa"
REPORT_DIR="${PROJECT_ROOT}/reports"

mkdir -p "${REPORT_DIR}"

# ── source Advisor environment ─────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${ADVISOR_HOME}/advisor-vars.sh" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate reports for one project
# Usage: gen_reports <project_dir> <label>
#
# Produces, under REPORT_DIR:
#   roofline_<label>.html    standalone interactive chart (--report=roofline)
#   topdown_<label>.csv      per-loop self-time + AI + GFLOPs (--report=top-down)
#   tripcounts_<label>.csv   per-loop trip counts + FLOP / byte traffic
#   survey_<label>.txt       text survey report
#
# NOTE on --format: `--report=roofline` ALWAYS writes an HTML chart and ignores
# --format.  For machine-readable per-loop data, use --report=top-down or
# --report=tripcounts with --format=csv.
# ─────────────────────────────────────────────────────────────────────────────
gen_reports() {
    local proj="$1"
    local label="$2"

    local html_out="${REPORT_DIR}/roofline_${label}.html"
    local topdown_out="${REPORT_DIR}/topdown_${label}.csv"
    local trip_out="${REPORT_DIR}/tripcounts_${label}.csv"
    local survey_out="${REPORT_DIR}/survey_${label}.txt"

    echo ""
    echo "── Generating reports: ${label} ──"

    # Interactive HTML roofline chart.
    "${ADVISOR_BIN}" \
        --report=roofline \
        --project-dir="${proj}" \
        --report-output="${html_out}" \
        2>&1 | tail -5
    echo "  Roofline HTML  → ${html_out}"

    # Per-loop top-down CSV: ID, function/loop, self-time%, AI, GFLOPs achieved.
    "${ADVISOR_BIN}" \
        --report=top-down \
        --project-dir="${proj}" \
        --report-output="${topdown_out}" \
        --format=csv \
        2>&1 | tail -5
    echo "  Top-down CSV   → ${topdown_out}"

    # Per-loop trip counts + FLOP + cache-simulated byte traffic.
    "${ADVISOR_BIN}" \
        --report=tripcounts \
        --project-dir="${proj}" \
        --report-output="${trip_out}" \
        --format=csv \
        2>&1 | tail -5
    echo "  Tripcounts CSV → ${trip_out}"

    # Human-readable survey: hot loops ranked by self-time.
    "${ADVISOR_BIN}" \
        --report=survey \
        --project-dir="${proj}" \
        --report-output="${survey_out}" \
        --format=text \
        2>&1 | tail -5
    echo "  Survey text    → ${survey_out}"
}

gen_reports "${PROJ_STD}"  "standard"
gen_reports "${PROJ_NUMA}" "numa"

# ─────────────────────────────────────────────────────────────────────────────
# Quick comparison: extract top-10 loops from each CSV and print side by side
# ─────────────────────────────────────────────────────────────────────────────
COMP="${REPORT_DIR}/roofline_comparison.txt"
{
    echo "======================================================================"
    echo "  Cache-Aware Roofline Comparison"
    echo "  Standard path  vs  NUMA tensor-parallel path"
    echo "======================================================================"
    echo ""
    echo "=== TOP LOOPS — STANDARD (no NUMA) ==="
    # Print header + first 10 data rows of the top-down CSV
    head -1  "${REPORT_DIR}/topdown_standard.csv"
    tail -n +2 "${REPORT_DIR}/topdown_standard.csv" | head -10
    echo ""
    echo "=== TOP LOOPS — NUMA-TP ==="
    head -1  "${REPORT_DIR}/topdown_numa.csv"
    tail -n +2 "${REPORT_DIR}/topdown_numa.csv" | head -10
    echo ""
    echo "How to read the roofline:"
    echo "  - Low AI  + near DRAM BW ceiling  → DRAM-bound (remote DRAM = bad)"
    echo "  - Higher AI + near L3/L2 ceiling  → cache-bound (good for NUMA-TP)"
    echo "  - Column 'Self Time%' shows which loops dominate runtime."
    echo "======================================================================"
} | tee "${COMP}"

echo ""
echo "Reports written to: ${REPORT_DIR}/"
echo "Open full interactive GUI:"
echo "  ${ADVISOR_HOME}/bin64/advixe-gui ${PROJ_STD}"
echo "  ${ADVISOR_HOME}/bin64/advixe-gui ${PROJ_NUMA}"
