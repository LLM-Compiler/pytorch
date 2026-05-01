#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_advisor_reports.sh
#
# Exports roofline and loop-performance data from the two Advisor project
# directories created by run_advisor_roofline.sh.
#
# Outputs (under advisor_projects/reports/):
#   roofline_standard.csv   — per-loop AI + time for standard path
#   roofline_numa.csv       — per-loop AI + time for NUMA-TP path
#   survey_standard.txt     — human-readable survey report (standard)
#   survey_numa.txt         — human-readable survey report (NUMA-TP)
#   roofline_comparison.txt — side-by-side summary printed to terminal
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
# Usage: gen_reports <project_dir> <label> <csv_out> <survey_txt_out>
# ─────────────────────────────────────────────────────────────────────────────
gen_reports() {
    local proj="$1"
    local label="$2"
    local csv_out="$3"
    local txt_out="$4"

    echo ""
    echo "── Generating reports: ${label} ──"

    # Roofline CSV: loop name, time %, self-time, AI (FLOPs/Byte),
    # GFLOPs/s achieved, cache simulation traffic per memory level.
    "${ADVISOR_BIN}" \
        --report=roofline \
        --project-dir="${proj}" \
        --report-output="${csv_out}" \
        --format=csv \
        2>&1 | tail -5

    echo "  Roofline CSV  → ${csv_out}"

    # Survey text report: hot loops ranked by self-time.
    "${ADVISOR_BIN}" \
        --report=survey \
        --project-dir="${proj}" \
        --report-output="${txt_out}" \
        --format=text \
        2>&1 | tail -5

    echo "  Survey text   → ${txt_out}"
}

gen_reports "${PROJ_STD}"  "standard" \
    "${REPORT_DIR}/roofline_standard.csv" \
    "${REPORT_DIR}/survey_standard.txt"

gen_reports "${PROJ_NUMA}" "numa_tp" \
    "${REPORT_DIR}/roofline_numa.csv" \
    "${REPORT_DIR}/survey_numa.txt"

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
    # Print header + first 10 data rows of the CSV
    head -1  "${REPORT_DIR}/roofline_standard.csv"
    tail -n +2 "${REPORT_DIR}/roofline_standard.csv" | head -10
    echo ""
    echo "=== TOP LOOPS — NUMA-TP ==="
    head -1  "${REPORT_DIR}/roofline_numa.csv"
    tail -n +2 "${REPORT_DIR}/roofline_numa.csv" | head -10
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
