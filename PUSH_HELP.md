# How to allow pushes to GitHub from this workspace

This environment can edit and commit the repository locally, but pushing to GitHub
needs a working remote, credentials, and network access. If you want me to push
future changes, follow these steps.

## 1) Ensure the remote is configured
Check that `origin` points to your GitHub repo:

```bash
git remote -v
```

If it is missing or wrong, set it:

```bash
git remote add origin https://github.com/ibg-sketch/smart-money-futures-bot.git
# or, if origin already exists
git remote set-url origin https://github.com/ibg-sketch/smart-money-futures-bot.git
```

## 2) Provide credentials (GitHub username + Personal Access Token)
When push prompts for auth, use your GitHub username and a PAT with `repo` scope.
You can also preconfigure credentials so pushes are non-interactive:

```bash
git config --global credential.helper store
# then run one push; Git will cache the token
```

If SSH is preferred and allowed by your network, add an SSH key to your GitHub
account and set the remote to the SSH URL.

## 3) Verify network/proxy access
Previous attempts hit a `CONNECT 403` tunnel error, which means the proxy blocked
HTTPS to GitHub. Push from a network that allows `github.com:443` or adjust the
proxy rules to permit it. If that is impossible here, you can pull the branch to
another machine with access and push from there:

```bash
git pull
# on a machine with access
```

## 4) Push the current branch
Once remote, credentials, and network are OK:

```bash
git push -u origin work  # or your active branch
```

After a successful push, I can continue making changes and prepare pull requests
against the GitHub repository.
