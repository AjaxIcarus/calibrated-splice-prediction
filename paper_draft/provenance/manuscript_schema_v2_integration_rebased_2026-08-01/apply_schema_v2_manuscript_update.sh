#!/usr/bin/env bash
set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(pwd)"

current_draft="paper_draft/full_draft_current.md"
current_plan="paper_draft/figure_table_plan.md"

expected_draft_sha="1e4301fed2a2366e6a10f6c66307f3ca6ef8a6f105f5f1153fabf3041d03bd4f"
expected_plan_sha="457b244e15f06c799a5a9619beaa4ebf9cf571bc987d9a9232a5101c5ebe4910"

versioned_draft="paper_draft/full_draft_schema_v2_2026-08-01.md"
versioned_plan="paper_draft/figure_table_plan_schema_v2_2026-08-01.md"
backup_draft="paper_draft/full_draft_current.md.bak_pre_schema_v2_20260801"
backup_plan="paper_draft/figure_table_plan.md.bak_pre_schema_v2_20260801"
figure_seed11="figures/flank400_schema_v2_reliability_seed11.png"
figure_seed23="figures/flank400_schema_v2_reliability_seed23.png"
installed_generator="scripts/make_schema_v2_manuscript.py"

verify_sha256() {
    local path="$1"
    local expected="$2"
    local actual

    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: SHA-256 mismatch: $path"
        echo "expected: $expected"
        echo "actual:   $actual"
        exit 1
    fi
}

if [[ ! -f "$repo_dir/README.md" ]] || [[ ! -d "$repo_dir/paper_draft" ]]; then
    echo "ERROR: run this from the calibrated-splice-prediction repository root"
    exit 1
fi

(cd "$package_dir" && sha256sum -c SHA256SUMS)

verify_sha256 "$current_draft" "$expected_draft_sha"
verify_sha256 "$current_plan" "$expected_plan_sha"

for path in \
    "$versioned_draft" \
    "$versioned_plan" \
    "$backup_draft" \
    "$backup_plan" \
    "$figure_seed11" \
    "$figure_seed23" \
    "$installed_generator"
do
    if [[ -e "$path" ]]; then
        echo "ERROR: refusing to overwrite existing path: $path"
        exit 1
    fi
done

mkdir -p paper_draft figures scripts

cp "$current_draft" "$backup_draft"
cp "$current_plan" "$backup_plan"

cp "$package_dir/full_draft_schema_v2_2026-08-01.md" "$versioned_draft"
cp "$package_dir/figure_table_plan_schema_v2_2026-08-01.md" "$versioned_plan"
cp "$package_dir/figures/flank400_schema_v2_reliability_seed11.png" "$figure_seed11"
cp "$package_dir/figures/flank400_schema_v2_reliability_seed23.png" "$figure_seed23"
cp "$package_dir/make_schema_v2_manuscript.py" "$installed_generator"

cp "$versioned_draft" "$current_draft"
cp "$versioned_plan" "$current_plan"

payload_draft_sha="$(sha256sum "$package_dir/full_draft_schema_v2_2026-08-01.md" | awk '{print $1}')"
payload_plan_sha="$(sha256sum "$package_dir/figure_table_plan_schema_v2_2026-08-01.md" | awk '{print $1}')"
verify_sha256 "$current_draft" "$payload_draft_sha"
verify_sha256 "$current_plan" "$payload_plan_sha"

if rg -n \
    '416,140,000|416,160,000|416,053,912|832\.107824|181\.749102|flank400_chromosome_transfer_cross_seed' \
    "$current_draft"
then
    echo "ERROR: obsolete pre-schema-v2 claim remains in current draft"
    exit 1
fi

echo "PASS: package hashes validated"
echo "PASS: expected pre-schema-v2 manuscript state validated"
echo "PASS: backups and versioned manuscript files created"
echo "PASS: accepted seed-11 and seed-23 reliability figures installed"
echo "PASS: full_draft_current.md and figure_table_plan.md promoted to schema v2"
echo "PASS: Git was not modified"
