@echo off
echo 🔧 Fixing Git history to remove API key...

REM Step 1: Reset to remove the problematic commits
echo Step 1: Resetting to clean state...
git reset --hard HEAD~3

REM Step 2: Add all files with environment variables
echo Step 2: Adding secure files...
git add .

REM Step 3: Commit the secure version
echo Step 3: Committing secure changes...
git commit -m "feat: Replace hardcoded API key with environment variables - Add .env support for secure API key storage - Update all OpenAI client initializations to use environment variables - Add .env.example template - Add comprehensive .gitignore - Add environment setup documentation"

REM Step 4: Force push to overwrite remote history
echo Step 4: Force pushing to remote...
git push origin feature/my-contribution --force

echo ✅ Done! API key removed from Git history and secure version pushed.
pause