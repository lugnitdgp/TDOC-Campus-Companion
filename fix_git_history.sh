#!/bin/bash

echo "🔧 Fixing Git history to remove API key..."

# Step 1: Remove the API key from Git history
echo "Step 1: Removing API key from commit history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch core/faiss_retriever.py faiss_vector_store.py semantic_rag.py' \
  --prune-empty --tag-name-filter cat -- --all

# Step 2: Add the files back with environment variables
echo "Step 2: Adding files back with environment variables..."
git add core/faiss_retriever.py faiss_vector_store.py semantic_rag.py .env.example .gitignore ENVIRONMENT_SETUP.md

# Step 3: Commit the changes
echo "Step 3: Committing secure changes..."
git commit -m "feat: Replace hardcoded API key with environment variables

- Add .env support for secure API key storage
- Update all OpenAI client initializations to use environment variables
- Add .env.example template
- Add comprehensive .gitignore
- Add environment setup documentation"

# Step 4: Force push to overwrite remote history
echo "Step 4: Force pushing to remote..."
git push origin feature/my-contribution --force

echo "✅ Done! API key removed from Git history and secure version pushed."