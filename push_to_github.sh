#!/bin/bash
# Interactive script to push jetson-apriltag to GitHub

echo "=========================================="
echo "Push jetson-apriltag to GitHub"
echo "=========================================="
echo ""

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ GitHub username is required"
    exit 1
fi

cd ~/jetson_apriltag

# Check if repo exists (this will fail if it doesn't, which is fine)
echo ""
echo "⚠️  IMPORTANT: Make sure you've created the repository on GitHub first!"
echo "   Go to: https://github.com/new"
echo "   Repository name: jetson-apriltag"
echo "   DO NOT initialize with README (we already have one)"
echo ""
read -p "Press Enter after creating the repository, or Ctrl+C to cancel..."

# Remove existing remote
git remote remove origin 2>/dev/null || true

# Try SSH first (if keys are set up)
echo ""
echo "Attempting SSH method..."
git remote add origin "git@github.com:${GITHUB_USERNAME}/jetson-apriltag.git" 2>/dev/null
git branch -M main

if git push -u origin main 2>&1; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "   https://github.com/${GITHUB_USERNAME}/jetson-apriltag"
    exit 0
fi

# If SSH failed, try HTTPS
echo ""
echo "SSH failed, trying HTTPS..."
git remote remove origin
git remote add origin "https://github.com/${GITHUB_USERNAME}/jetson-apriltag.git"

echo ""
echo "Pushing via HTTPS (you'll be prompted for credentials)..."
echo "Note: Use a Personal Access Token as password (not your GitHub password)"
echo ""

if git push -u origin main; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "   https://github.com/${GITHUB_USERNAME}/jetson-apriltag"
else
    echo ""
    echo "❌ Push failed. Please check:"
    echo "   1. Repository exists at https://github.com/${GITHUB_USERNAME}/jetson-apriltag"
    echo "   2. You have push access"
    echo "   3. Your credentials are correct"
fi
