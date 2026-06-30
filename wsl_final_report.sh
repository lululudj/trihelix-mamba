#!/bin/bash
# Post-training: generate report + summary
cd /mnt/e/three_chain_v3
echo "=== Generating Final Report: $(date) ==="
python3 generate_report.py
echo "=== Done ==="

# Quick summary for terminal
echo ""
echo "============================================"
echo "  FULL BENCHMARK COMPLETE"
echo "============================================"
echo "Report: results_wsl/benchmark_report.json"
echo ""
echo "Checkpoints:"
echo "  Standard: $(ls results_wsl/benchmark_multiseed/*/metrics.json 2>/dev/null | wc -l)"
echo "  Scaled:   $(ls results_wsl/benchmark_scaled/*/metrics.json 2>/dev/null | wc -l)"
echo "  Ablation: $(ls results_wsl/ablation/*/metrics.json 2>/dev/null | wc -l)"
