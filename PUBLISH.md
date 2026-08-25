# Publish orbaxport to GitHub + PyPI

## 1. Create GitHub repo

1. Go to https://github.com/new
2. Name: `orbaxport` (public)
3. Do **not** add README/license (already in this folder)
4. Create repository

## 2. Push code

```bash
cd orbaxport_pkg   # this directory

git init
git add .
git commit -m "Initial release v1.1.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/orbaxport.git
git push -u origin main
```

Replace `YOUR_USERNAME` in `README.md` and `pyproject.toml` project.urls as well.

## 3. Install locally (sanity check)

```bash
pip install -e ".[dev]"
orbaxport --help
pytest tests/ -q
```

## 4. Build distributions

```bash
pip install build twine
python -m build
ls dist/
# orbaxport-1.1.0.tar.gz
# orbaxport-1.1.0-py3-none-any.whl
```

## 5. Upload to TestPyPI (recommended first)

```bash
# Create account: https://test.pypi.org/account/register/
# API token: https://test.pypi.org/manage/account/token/

python -m twine upload --repository testpypi dist/*

# Test install
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple orbaxport
```

## 6. Upload to real PyPI

```bash
# Account: https://pypi.org/account/register/
# API token: https://pypi.org/manage/account/token/
# Trusted Publisher (OIDC) optional via GitHub Actions – see .github/workflows/publish.yml

python -m twine upload dist/*
```

## 7. GitHub Release (triggers Actions publish if configured)

```bash
git tag v1.1.0
git push origin v1.1.0
# GitHub → Releases → Draft a new release from tag v1.1.0
```

For **Trusted Publishing** (no API token in secrets):

1. PyPI → Project → Settings → Publishing → Add GitHub workflow
2. Owner/repo: `YOUR_USERNAME/orbaxport`
3. Workflow: `publish.yml`
4. Environment: leave empty or set `pypi`

## 8. Bump version later

Edit in both places:

- `pyproject.toml` → `version`
- `src/orbaxport/__init__.py` → `__version__`

Then rebuild + tag + upload.
