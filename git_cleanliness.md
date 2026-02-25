# Git Cleanliness Guidelines

To avoid push rejections and messy rebase states, we will follow these protocols in the future.

## Root Causes of Recent Issues
1. **Race Conditions**: GitHub Actions pushes an "Update analysis results" commit while we are working locally, causing our push to be rejected.
2. **Context Contamination**: Committing large, frequently-changing data files (`web_data.json`, `.csv` files) alongside code causes frequent merge conflicts.
3. **Rebase Lock**: Multiple background terminal sessions running `git rebase --continue` occurred, likely due to unintended tool usage or session persistence.

## Protocol for Clean Pushing
1. **Fetch & Pull First**: Always run `git pull --rebase origin master` *before* starting a new round of edits.
2. **Surgical Commits**: Commit only the code files (`.py`, `.html`, `.css`, `.md`). Avoid committing generated data files unless specifically asked.
3. **The Stash Sequence**:
    - `git stash push -m "local code backup"`
    - `git pull --rebase origin master`
    - `git push origin master`
    - `git stash pop`
4. **Data Conflict Resolution**:
    - If `stash pop` or `pull` fails due to conflicts in `web_data.json` or `.csv` files:
    - Run: `git checkout --theirs output/rs_industries.csv output/rs_setups.csv output/rs_stocks.csv output/web_data.json data_persist/ticker_info.json`
    - Run: `git add .` to mark resolution, then continue.
    - This ensures code always moves forward while data conflicts are ignored.
5. **Verification**: Always run `git status` before `git push` to ensure NO data files are staged for the remote.
