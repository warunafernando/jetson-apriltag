# Pushing to GitHub

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `jetson-apriltag`
3. Description: "CUDA-accelerated AprilTag detection for Jetson using Team 971/766 detector"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Add Remote and Push

```bash
cd ~/jetson_apriltag

# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/jetson-apriltag.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/jetson-apriltag.git

# Push to GitHub
git branch -M main  # Rename master to main (GitHub standard)
git push -u origin main
```

## Step 3: Verify

Visit `https://github.com/YOUR_USERNAME/jetson-apriltag` to verify all files are uploaded.

## Future Updates

```bash
cd ~/jetson_apriltag
git add .
git commit -m "Your commit message"
git push
```
