#!/bin/bash
# Script to push jetson-apriltag to GitHub
# Replace YOUR_USERNAME with your GitHub username before running

GITHUB_USERNAME="${1:-YOUR_USERNAME}"

if [ "$GITHUB_USERNAME" = "YOUR_USERNAME" ]; then
    echo "Usage: $0 <your-github-username>"
    echo ""
    echo "Example: $0 nav"
    exit 1
fi

cd ~/jetson_apriltag

# Remove existing remote if any
git remote remove origin 2>/dev/null || true

# Add remote
git remote add origin "https://github.com/${GITHUB_USERNAME}/jetson-apriltag.git"

# Rename branch to main
git branch -M main

# Push
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Pushed to: https://github.com/${GITHUB_USERNAME}/jetson-apriltag"
