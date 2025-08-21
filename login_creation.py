import os, subprocess, shlex

def run(cmd, cwd=None):
    r = subprocess.run(shlex.split(cmd), cwd=cwd, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout.strip()

def git_push(repo_dir, message="update", branch="main", files=None):
    token = os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN manquant.")
    # Config git
    run("git config user.name bot")
    run("git config user.email bot@example.invalid")
    run(f"git -C {repo_dir} config --global --add safe.directory {repo_dir}")

    # Stage
    if files:
        run(f"git -C {repo_dir} add " + " ".join(files))
    else:
        run(f"git -C {repo_dir} add -A")

    # Rien à committer ?
    code = subprocess.run(shlex.split(f"git -C {repo_dir} diff --cached --quiet")).returncode
    if code == 0:
        return "Rien à committer."

    run(f'git -C {repo_dir} commit -m "{message}"')

    # Récupère l’URL d’origine et insère le token
    origin = run(f"git -C {repo_dir} remote get-url origin")
    if origin.startswith("https://"):
        authed = origin.replace("https://", f"https://{token}@")
    elif origin.startswith("git@github.com:"):
        owner_repo = origin.split(":", 1)[1].rstrip(".git")
        authed = f"https://{token}@github.com/{owner_repo}.git"
    else:
        raise RuntimeError(f"Remote non reconnu: {origin}")

    run(f"git -C {repo_dir} push {authed} HEAD:{branch}")
    return "✅ Push effectué."

git_push("C:/Users\hugom\OneDrive\Documents\Git_Stage_2025_ZMH", message="update", branch="main", files="C:/Users\hugom\OneDrive\Documents\Git_Stage_2025_ZMH\data/trends.json")