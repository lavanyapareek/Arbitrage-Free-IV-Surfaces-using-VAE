# Setup Guide for GitHub Repository

## BEFORE PUSHING TO GITHUB

### Step 1: Remove Sensitive Files

**CRITICAL**: These files contain secrets and must be removed:

```bash
# Remove API key file (will use template instead)
git rm --cached api.json

# Remove Google Cloud credentials
git rm --cached gdeltplaypal-be8da892c655.json

# If you accidentally committed them, you MUST revoke the keys:
# 1. Gemini API: https://aistudio.google.com/app/apikey (delete and create new)
# 2. GCP Service Account: https://console.cloud.google.com/iam-admin/serviceaccounts
```

### Step 2: Handle Large Files

GitHub has a 100 MB file size limit. Check large files:

```bash
# Find files larger than 50 MB
find . -type f -size +50M

# Options for large files:
# 1. Add to .gitignore (recommended for generated data)
# 2. Use Git LFS (for essential large files)
# 3. Host externally (Google Drive, Zenodo, etc.)
```

**Large files in your project:**
- `final_options_data_filled.csv` (235 MB) - **EXCLUDE**
- `nifty_advanced_surfaces.pickle` (3.5 MB) - Consider Git LFS
- `nifty_filtered_surfaces.pickle` (900 KB) - OK to include
- `vae_single_param/` directory - **EXCLUDE** (training artifacts)

### Step 3: Clean Up Generated Files

```bash
# Remove generated results
rm -rf notebook_results/
rm -rf demo_results/
rm -rf condtional_vae/results_date/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
```

### Step 4: Verify .gitignore

Ensure `.gitignore` is working:

```bash
# Check what will be committed
git status

# Verify sensitive files are ignored
git check-ignore api.json
git check-ignore gdeltplaypal-be8da892c655.json

# Should output the filenames if properly ignored
```

### Step 5: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files (respecting .gitignore)
git add .

# Check what's staged
git status

# Commit
git commit -m "Initial commit: Arbitrage-Free IV Surface Generation using CVAE"
```

### Step 6: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (public or private)
3. **DO NOT** initialize with README (you already have one)

### Step 7: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push
git branch -M main
git push -u origin main
```

## Pre-Push Checklist

- [ ] `.gitignore` file created and configured
- [ ] `api.json` removed (template provided)
- [ ] `gdeltplaypal-*.json` removed and key revoked
- [ ] Large files (>100 MB) excluded or using Git LFS
- [ ] Generated results folders excluded
- [ ] Python cache files removed
- [ ] README.md created with setup instructions
- [ ] requirements.txt is up to date
- [ ] Sensitive information removed from notebooks
- [ ] License file added (if applicable)

## Security Best Practices

### For Users Cloning Your Repo

Add this to README:

```markdown
## Setup for New Users

1. Clone the repository
2. Copy `api.json.template` to `api.json`
3. Add your Gemini API key to `api.json`
4. Never commit `api.json` to version control
```

### Rotating Compromised Keys

If you accidentally pushed secrets:

1. **Immediately revoke the keys**:
   - Gemini API: Delete key at https://aistudio.google.com/app/apikey
   - GCP: Disable service account at GCP Console

2. **Remove from Git history** (if already pushed):
```bash
# Use BFG Repo-Cleaner or git-filter-repo
git filter-repo --path api.json --invert-paths
git filter-repo --path gdeltplaypal-be8da892c655.json --invert-paths

# Force push (WARNING: rewrites history)
git push origin --force --all
```

3. **Create new keys** and update locally

## Optional: Git LFS for Large Files

If you need to include large model files:

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: sudo apt-get install git-lfs  # Linux

# Initialize
git lfs install

# Track large files
git lfs track "*.pickle"
git lfs track "*.pt"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

## Alternative: External Data Hosting

For very large files, consider:

1. **Zenodo** (academic datasets): https://zenodo.org
2. **Google Drive** (with public link)
3. **Hugging Face Datasets**: https://huggingface.co/datasets
4. **AWS S3** (with public bucket)

Add download instructions to README:

```markdown
## Data Files

Large data files are hosted externally:

1. Download from: [link]
2. Extract to project root
3. Verify file structure matches documentation
```

## Final Verification

Before pushing:

```bash
# Dry run to see what would be pushed
git push --dry-run

# Check repository size
du -sh .git

# Verify no secrets in history
git log --all --full-history --source -- api.json
git log --all --full-history --source -- "*credentials*.json"
```

## Troubleshooting

**Problem**: "File too large" error
- **Solution**: Add to .gitignore or use Git LFS

**Problem**: Accidentally committed secrets
- **Solution**: Revoke keys immediately, use git-filter-repo to clean history

**Problem**: Repository too large
- **Solution**: Remove large files, use Git LFS, or host externally

## Resources

- [GitHub .gitignore templates](https://github.com/github/gitignore)
- [Git LFS documentation](https://git-lfs.github.com/)
- [Removing sensitive data from Git](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
