#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(pwd)"

if [[ ! -f "$repo_dir/paper_draft/full_draft_current.md" ]]; then
    echo "ERROR: run from the calibrated-splice-prediction repository root"
    exit 1
fi

if [[ ! -f "$repo_dir/paper_draft/figure_table_plan.md" ]]; then
    echo "ERROR: missing paper_draft/figure_table_plan.md"
    exit 1
fi

(
    cd "$package_dir"
    sha256sum -c SHA256SUMS
)
echo "PASS: package hashes validated"

expected_manuscript_sha="25cebef9d33f0fb6c4d6e4eb2f998c07eec5f750e577036514d8b213235737f0"
expected_plan_sha="046a3582a6375f98bea33ce837a5bbd7daa41030e3f3cc4ccd68baab064f65ab"

actual_manuscript_sha="$(sha256sum "$repo_dir/paper_draft/full_draft_current.md" | awk '{print $1}')"
actual_plan_sha="$(sha256sum "$repo_dir/paper_draft/figure_table_plan.md" | awk '{print $1}')"

if [[ "$actual_manuscript_sha" != "$expected_manuscript_sha" ]]; then
    echo "ERROR: SHA-256 mismatch: paper_draft/full_draft_current.md"
    echo "expected: $expected_manuscript_sha"
    echo "actual:   $actual_manuscript_sha"
    exit 1
fi

if [[ "$actual_plan_sha" != "$expected_plan_sha" ]]; then
    echo "ERROR: SHA-256 mismatch: paper_draft/figure_table_plan.md"
    echo "expected: $expected_plan_sha"
    echo "actual:   $actual_plan_sha"
    exit 1
fi
echo "PASS: expected promoted schema-v2 manuscript state validated"

backup_dir="$repo_dir/paper_draft/backups/pre_polish_schema_v2_2026-08-01"
versioned_manuscript="$repo_dir/paper_draft/full_draft_schema_v2_polished_2026-08-01.md"
versioned_plan="$repo_dir/paper_draft/figure_table_plan_schema_v2_polished_2026-08-01.md"
supplement="$repo_dir/paper_draft/supplementary_materials_schema_v2_2026-08-01.md"
decision_note="$repo_dir/paper_draft/editorial_decisions_schema_v2_2026-08-01.md"
supplement_data_dir="$repo_dir/paper_draft/supplementary_data_schema_v2_2026-08-01"

for target in \
    "$backup_dir" \
    "$versioned_manuscript" \
    "$versioned_plan" \
    "$supplement" \
    "$decision_note" \
    "$supplement_data_dir"
do
    if [[ -e "$target" ]]; then
        echo "ERROR: refusing to overwrite existing target: $target"
        exit 1
    fi
done

mkdir -p "$backup_dir" "$supplement_data_dir"
cp "$repo_dir/paper_draft/full_draft_current.md" "$backup_dir/full_draft_current.md"
cp "$repo_dir/paper_draft/figure_table_plan.md" "$backup_dir/figure_table_plan.md"

cp "$package_dir/full_draft_schema_v2_polished_2026-08-01.md" "$versioned_manuscript"
cp "$package_dir/figure_table_plan_schema_v2_polished_2026-08-01.md" "$versioned_plan"
cp "$package_dir/supplementary_materials_schema_v2_2026-08-01.md" "$supplement"
cp "$package_dir/editorial_decisions_2026-08-01.md" "$decision_note"
cp "$package_dir/source_tables/"* "$supplement_data_dir/"
echo "PASS: backups and versioned polish files created"

cp "$versioned_manuscript" "$repo_dir/paper_draft/full_draft_current.md"
cp "$versioned_plan" "$repo_dir/paper_draft/figure_table_plan.md"
echo "PASS: polished manuscript and figure/table plan promoted"

if rg -n -i \
    'results/|data/processed|Comparison with flank-80|flank-80 analyses showed' \
    "$repo_dir/paper_draft/full_draft_current.md"
then
    echo "ERROR: internal path or retired flank-80 claim remains"
    exit 1
fi
echo "PASS: internal paths and retired flank-80 claims are absent"

if ! rg -q 'Supplementary Table S1' "$repo_dir/paper_draft/full_draft_current.md"; then
    echo "ERROR: supplementary interval citation missing"
    exit 1
fi

if ! rg -q 'None of the ten paired intervals excludes zero' "$supplement"; then
    echo "ERROR: paired-interval interpretation missing from supplement"
    exit 1
fi
echo "PASS: supplementary interval and reliability-bin placement validated"

if git diff --quiet -- .git 2>/dev/null; then
    :
fi
echo "PASS: Git was not modified"
