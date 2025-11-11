#!/usr/bin/env bash
set -euo pipefail

# Close open PRs older than 1 day. Usage:
#   ./close-stale-prs.sh       # dry run
#   ./close-stale-prs.sh --apply  # comment+close

DRY_RUN=1
if [ "${1:-}" = "--apply" ]; then
  DRY_RUN=0
fi

THRESHOLD_MS=$((24*60*60*1000))
TMP_JSON=/tmp/mlv_prs.json

echo "Fetching open PRs..."
if ! gh pr list --state open --json number,title,createdAt,url,author --limit 200 > "$TMP_JSON"; then
  echo "Failed to run 'gh pr list'. Is the GitHub CLI installed and authenticated?" >&2
  exit 1
fi

PRS_JSON=$(node -e 'const fs=require("fs"); const data=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); const now=Date.now(); const cutoff=now-24*60*60*1000; const out=[]; for(const pr of data){ if(new Date(pr.createdAt).getTime()<cutoff) out.push({number:pr.number,title:pr.title,createdAt:pr.createdAt,url:pr.url,author:(pr.author&&pr.author.login)||"unknown"}); } console.log(JSON.stringify(out));' "$TMP_JSON")

if [ -z "$PRS_JSON" ] || [ "$PRS_JSON" = "[]" ]; then
  echo "No PRs older than 1 day to close."
  exit 0
fi

# parse PRs and act
count=$(node -e 'const arr=JSON.parse(process.argv[1]); console.log(arr.length);' "$PRS_JSON")

echo "Found $count PR(s) older than 1 day."

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run mode — listing PRs that WOULD be closed:\n"
  node -e 'const arr=JSON.parse(process.argv[1]); for(const pr of arr){ console.log(`#${pr.number}\t${pr.createdAt}\t${pr.title}\t${pr.url}\t${pr.author}`); }' "$PRS_JSON"
  echo "\nRun with --apply to actually post comments and close these PRs."
  exit 0
fi

# Apply mode: comment and close each PR
for num in $(node -e 'const arr=JSON.parse(process.argv[1]); for(const pr of arr) console.log(pr.number);' "$PRS_JSON"); do
  echo "Commenting PR #$num"
  gh pr comment "$num" --body "Closing this PR as stale (no activity for >1 day). Please reopen or open a fresh PR if you'd like to continue this work. Thanks!" || echo "Failed to comment on PR #$num"
  echo "Closing PR #$num"
  gh pr close "$num" || echo "Failed to close PR #$num"
done

echo "Done."
