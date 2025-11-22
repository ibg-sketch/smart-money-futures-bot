# How to allow pushes to GitHub from this workspace

This environment can edit and commit the repository locally, but pushing to GitHub
needs a working remote, credentials, and network access. To let me push new
commits directly to GitHub, run through this short checklist.

### Quick answer (what you need to do)
1) Remote уже настроен на `https://github.com/ibg-sketch/smart-money-futures-bot.git` — оставьте его так или подтвердите через `git remote -v`.
2) Дайте доступ: используйте GitHub username + Personal Access Token (scope `repo`) при первом пуше; можно включить `credential.helper store`, чтобы не вводить токен каждый раз.
3) Проверьте сеть: ошибка `CONNECT 403` означает, что прокси блокирует GitHub. Нужно разблокировать `github.com:443`, использовать VPN/другую сеть или пушить с машины, у которой есть доступ.
4) Выполните `git push -u origin work` (или другую активную ветку), чтобы подтвердить доступ. После этого я смогу пушить сам.

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
