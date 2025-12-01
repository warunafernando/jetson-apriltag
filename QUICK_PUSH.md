# Quick Push to GitHub

## Option 1: Using GitHub CLI (if installed)

```bash
cd ~/jetson_apriltag
gh repo create jetson-apriltag --public --source=. --remote=origin --push
```

## Option 2: Manual Steps

### Step 1: Create Repository on GitHub
1. Go to https://github.com/new
2. Repository name: `jetson-apriltag`
3. Description: "CUDA-accelerated AprilTag detection for Jetson"
4. Choose Public or Private
5. **DO NOT** check "Add a README file" (we already have one)
6. Click "Create repository"

### Step 2: Push (choose one method)

#### Method A: HTTPS (requires personal access token)
```bash
cd ~/jetson_apriltag
git remote remove origin 2>/dev/null
git remote add origin https://github.com/YOUR_USERNAME/jetson-apriltag.git
git branch -M main
git push -u origin main
# When prompted, use your GitHub username and a Personal Access Token as password
```

#### Method B: SSH (if you have SSH keys set up)
```bash
cd ~/jetson_apriltag
git remote remove origin 2>/dev/null
git remote add origin git@github.com:YOUR_USERNAME/jetson-apriltag.git
git branch -M main
git push -u origin main
```

### Step 3: Or use the helper script
```bash
cd ~/jetson_apriltag
bash push_commands.sh YOUR_USERNAME
```

Replace `YOUR_USERNAME` with your actual GitHub username.
