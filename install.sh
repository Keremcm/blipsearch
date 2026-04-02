#!/bin/bash
set -e

echo "✨ Installing Blipsearch..."

# Check root status (warn if running with sudo)
if [ "$EUID" -eq 0 ]; then
  echo "⚠️ Warning: It is not recommended to run this script as root."
  echo "Please run it as a normal user."
  exit 1
fi

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

INSTALL_DIR="$HOME/.local/share/blipsearch-env"
BIN_DIR="$HOME/.local/bin"

echo "📦 Setting up isolated environment..."
python3 -m venv "$INSTALL_DIR"

echo "📥 Installing dependencies (this might take a few minutes)..."
"$INSTALL_DIR/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/bin/pip" install --quiet -e .

echo "🔗 Registering blipsearch command..."
mkdir -p "$BIN_DIR"
# Create a symlink to make it available universally
ln -sf "$INSTALL_DIR/bin/blipsearch" "$BIN_DIR/blipsearch"

echo ""
echo "✅ Installation strictly completed!"
echo "🚀 You can now type 'blipsearch' from anywhere in your terminal."
echo ""
echo "Note: If 'blipsearch' command is not found, you must restart your terminal"
echo "or run this command once to update your PATH:"
echo 'export PATH="$HOME/.local/bin:$PATH"'
