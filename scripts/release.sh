#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== @sudocode-ai/claude-code-acp Release Script ===${NC}\n"

# Check if we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
  echo -e "${YELLOW}Warning: You're on branch '$BRANCH', not 'main'${NC}"
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Check for uncommitted changes (ignoring untracked files)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo -e "${RED}Error: You have uncommitted changes. Please commit or stash them first.${NC}"
  git status --short --untracked-files=no
  exit 1
fi

# Pull latest changes
echo -e "${YELLOW}Pulling latest changes...${NC}"
git pull origin "$BRANCH"

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
npm run test:run

# Build
echo -e "${YELLOW}Building...${NC}"
npm run build

# Show current version
CURRENT_VERSION=$(node -p "require('./package.json').version")
echo -e "\nCurrent version: ${GREEN}$CURRENT_VERSION${NC}"

# Ask for version bump type
echo -e "\nSelect version bump type:"
echo "  1) patch (bug fixes)"
echo "  2) minor (new features, backwards compatible)"
echo "  3) major (breaking changes)"
echo "  4) custom (enter version manually)"
echo "  5) cancel"

read -p "Choice [1-5]: " -n 1 -r
echo

case $REPLY in
  1)
    VERSION_TYPE="patch"
    ;;
  2)
    VERSION_TYPE="minor"
    ;;
  3)
    VERSION_TYPE="major"
    ;;
  4)
    read -p "Enter version (e.g., 1.2.3): " CUSTOM_VERSION
    VERSION_TYPE="custom"
    ;;
  5|*)
    echo "Release cancelled."
    exit 0
    ;;
esac

# Calculate new version
if [ "$VERSION_TYPE" = "custom" ]; then
  NEW_VERSION="$CUSTOM_VERSION"
else
  NEW_VERSION=$(npm version "$VERSION_TYPE" --no-git-tag-version | sed 's/v//')
  # Reset the change since npm version will be called again
  git checkout package.json package-lock.json 2>/dev/null || git checkout package.json
fi

echo -e "\nNew version will be: ${GREEN}$NEW_VERSION${NC}"

# Confirm release
echo -e "\n${YELLOW}This will:${NC}"
echo "  1. Bump version to $NEW_VERSION"
echo "  2. Create a git commit and tag"
echo "  3. Push to origin"
echo "  4. Publish to npm"

read -p "Continue with release? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Release cancelled."
  exit 0
fi

# Bump version (this runs build via npm scripts)
echo -e "\n${YELLOW}Bumping version...${NC}"
if [ "$VERSION_TYPE" = "custom" ]; then
  npm version "$CUSTOM_VERSION" -m "chore: release v%s"
else
  npm version "$VERSION_TYPE" -m "chore: release v%s"
fi

# Push to origin
echo -e "\n${YELLOW}Pushing to origin...${NC}"
git push origin "$BRANCH" --tags

# Publish to npm
echo -e "\n${YELLOW}Publishing to npm...${NC}"
npm publish --access public

echo -e "\n${GREEN}=== Release v$NEW_VERSION complete! ===${NC}"
echo -e "View on npm: https://www.npmjs.com/package/@sudocode-ai/claude-code-acp"
