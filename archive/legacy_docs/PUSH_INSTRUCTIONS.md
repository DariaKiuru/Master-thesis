# Push instructions

I created a `.gitignore` to exclude virtualenv and common temporary files.

You already have a commit in the repo: `Save: commit via assistant`.

To finish cleanup and push to `origin`, run these commands locally (PowerShell):

```powershell
git status
git add .gitignore
git rm -r --cached .venv || git rm -r --cached venv || true
git commit -m "Remove venv from repo and add .gitignore"
git push origin HEAD
```

If `git` is not installed, install it from https://git-scm.com/ and ensure it's on your PATH.

Authentication notes:
- Use the Git Credential Manager or an SSH key for seamless pushes.
- If prompted for username/password, use a personal access token (PAT) as password for HTTPS pushes.

If you'd like, enable `git` in this environment and I will retry the `git push` for you.
