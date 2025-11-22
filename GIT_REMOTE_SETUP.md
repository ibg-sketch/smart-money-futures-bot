# GitHub Remote Setup

This repository did not have a Git remote configured by default. To push the current branch to GitHub:

1. Add the remote (already set to `origin`):
   ```bash
   git remote add origin https://github.com/ibg-sketch/smart-money-futures-bot.git
   ```

2. Verify the remote:
   ```bash
   git remote -v
   ```

3. Push the current branch (replace `work` with another branch name if needed):
   ```bash
   git push -u origin work
   ```

If the remote already exists and you need to update the URL, run:
```bash
git remote set-url origin https://github.com/ibg-sketch/smart-money-futures-bot.git
```
