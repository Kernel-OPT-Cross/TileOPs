#!/usr/bin/env bash
set -uo pipefail

output_dir="${1:-sm120-test-results}"
logs_dir="${output_dir}/logs"
summary="${output_dir}/summary.tsv"

mkdir -p "${logs_dir}"
printf 'status\ttest_file\texit_code\n' > "${summary}"

total=0
passed=0
failed=0

while IFS= read -r test_file; do
    total=$((total + 1))
    log_name="${test_file//\//_}.log"

    echo "[${total}] ${test_file}"
    python -m pytest -ra -q "${test_file}" > "${logs_dir}/${log_name}" 2>&1
    exit_code=$?

    if [[ ${exit_code} -eq 0 ]]; then
        status=PASS
        passed=$((passed + 1))
    else
        status=FAIL
        failed=$((failed + 1))
    fi

    printf '%s\t%s\t%s\n' "${status}" "${test_file}" "${exit_code}" >> "${summary}"
done < <(find tests -type f -name 'test_*.py' | sort)

printf 'TOTAL\t%s\nPASS\t%s\nFAIL\t%s\n' "${total}" "${passed}" "${failed}" | tee "${output_dir}/file-totals.txt"
touch "${output_dir}/DONE"

[[ ${failed} -eq 0 ]]
