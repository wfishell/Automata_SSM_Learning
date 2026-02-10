#!/bin/bash
# Script to filter TLSF files:
# - Removes files that take longer than 5 seconds to synthesize
# - Removes files that are unrealizable
#
# Usage: ./filter_tlsf.sh [--dry-run] [--timeout SECONDS]
#   --dry-run    Don't actually delete files, just report what would be deleted
#   --timeout N  Set timeout to N seconds (default: 5)

TIMEOUT_SECONDS=5
TLSF_ROOT="/workspaces/Automata_SSM_Learning/TestSet/benchmarks/tlsf"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--timeout SECONDS]"
            exit 1
            ;;
    esac
done

# Check if ltlsynt is available
if ! command -v ltlsynt &> /dev/null; then
    echo "ERROR: ltlsynt not found in PATH"
    echo "Please install the Spot library (https://spot.lrde.epita.fr/)"
    exit 1
fi

echo "Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN (no files will be deleted)" || echo "LIVE (files will be deleted)")"
echo ""

# Counters for statistics
total_files=0
removed_timeout=0
removed_unrealizable=0
kept_files=0

# Log file for removed files
LOG_FILE="$(dirname "$TLSF_ROOT")/filter_log_$(date +%Y%m%d_%H%M%S).txt"

echo "TLSF Filter Log - $(date)" > "$LOG_FILE"
echo "Timeout: ${TIMEOUT_SECONDS} seconds" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Find all .tlsf files recursively
while IFS= read -r -d '' tlsf_file; do
    total_files=$((total_files + 1))

    echo -n "Processing: $tlsf_file ... "

    # Run ltlsynt with timeout
    # Capture both stdout and stderr, and the exit code
    start_time=$(date +%s.%N)

    output=$(timeout ${TIMEOUT_SECONDS}s ltlsynt --tlsf "$tlsf_file" 2>&1)
    exit_code=$?

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    # Check if timeout occurred (exit code 124)
    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT (>${TIMEOUT_SECONDS}s) - $([ "$DRY_RUN" = true ] && echo "WOULD REMOVE" || echo "REMOVING")"
        echo "TIMEOUT: $tlsf_file (>${TIMEOUT_SECONDS}s)" >> "$LOG_FILE"
        [ "$DRY_RUN" = false ] && rm "$tlsf_file"
        removed_timeout=$((removed_timeout + 1))
        continue
    fi

    # Check if unrealizable
    if echo "$output" | grep -q "UNREALIZABLE"; then
        echo "UNREALIZABLE - $([ "$DRY_RUN" = true ] && echo "WOULD REMOVE" || echo "REMOVING")"
        echo "UNREALIZABLE: $tlsf_file" >> "$LOG_FILE"
        [ "$DRY_RUN" = false ] && rm "$tlsf_file"
        removed_unrealizable=$((removed_unrealizable + 1))
        continue
    fi

    # Check for other errors (ltlsynt not found, parse errors, etc.)
    if [ $exit_code -ne 0 ]; then
        echo "ERROR (exit code $exit_code) - KEEPING (manual review needed)"
        echo "ERROR ($exit_code): $tlsf_file" >> "$LOG_FILE"
        kept_files=$((kept_files + 1))
        continue
    fi

    # File is realizable and synthesized within timeout
    printf "OK (%.2fs)\n" "$elapsed"
    kept_files=$((kept_files + 1))

done < <(find "$TLSF_ROOT" -name "*.tlsf" -type f -print0)

# Print summary
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total files processed: $total_files"
echo "Kept (realizable, <${TIMEOUT_SECONDS}s): $kept_files"
echo "Removed (timeout): $removed_timeout"
echo "Removed (unrealizable): $removed_unrealizable"
echo "========================================"
echo "Log saved to: $LOG_FILE"

# Also append summary to log
echo "" >> "$LOG_FILE"
echo "========================================"  >> "$LOG_FILE"
echo "SUMMARY" >> "$LOG_FILE"
echo "Total files: $total_files" >> "$LOG_FILE"
echo "Kept: $kept_files" >> "$LOG_FILE"
echo "Removed (timeout): $removed_timeout" >> "$LOG_FILE"
echo "Removed (unrealizable): $removed_unrealizable" >> "$LOG_FILE"
