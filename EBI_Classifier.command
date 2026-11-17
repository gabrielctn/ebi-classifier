#!/bin/bash
# =============================================================================
# EBI Classifier Launcher (Double-click to run)
# =============================================================================
# This script automatically sets up and launches the EBI classification app.
# Just double-click this file in Finder to start!
# =============================================================================

# Go to the folder where this script lives
cd "$(dirname "$0")"
export PATH="$PWD/.ebi-runtime:$PATH"

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              🔬 EBI Classifier - Starting...             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check if uv (Python package manager) is installed
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}📦 First-time setup: Installing uv (Python package manager)...${NC}"
    echo "   This only needs to happen once."
    echo ""
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        echo ""
        echo "❌ Could not install uv automatically."
        echo "   Please close this window, restart Terminal, and try again."
        echo ""
        echo "Press Enter to close..."
        read
        exit 1
    fi
    echo -e "${GREEN}✅ uv installed successfully!${NC}"
    echo ""
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ---------------------------------------------------------------------------
# Step 2: Create virtual environment if it doesn't exist
# ---------------------------------------------------------------------------
if [ ! -d ".venv-secure" ]; then
    echo -e "${YELLOW}📦 First-time setup: Creating Python environment...${NC}"
    echo "   This may take a few minutes."
    echo ""
    
    ./setup.sh
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Failed to create Python environment."
        echo "   Please make sure you have an internet connection."
        echo ""
        echo "Press Enter to close..."
        read
        exit 1
    fi
    echo -e "${GREEN}✅ Python environment created!${NC}"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: Install/update dependencies
# ---------------------------------------------------------------------------
# Activate the virtual environment
source .venv-secure/bin/activate

# Do not silently run the source launcher with an interpreter older than the
# security releases used by the updated packages.
if ! python -c 'import sys; p = sys.version_info[:3]; minimum = {(3, 11): (3, 11, 18), (3, 12): (3, 12, 16), (3, 13): (3, 13, 18), (3, 14): (3, 14, 10), (3, 15): (3, 15, 2)}; sys.exit(0 if p[:2] in minimum and p >= minimum[p[:2]] else 1)'; then
    echo "The local development environment uses an outdated Python interpreter."
    echo "Move .venv-secure to a backup name, then run ./setup.sh, or use the dated Mac ZIP."
    echo "Press Enter to close..."
    read
    exit 1
fi

# Check if dependencies are installed by looking for gradio
if ! python -c "import gradio" &> /dev/null; then
    echo -e "${YELLOW}📦 Installing required packages...${NC}"
    echo "   This may take a few minutes on first run."
    echo ""
    
    ./setup.sh
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Failed to install packages."
        echo "   Please make sure you have an internet connection."
        echo ""
        echo "Press Enter to close..."
        read
        exit 1
    fi
    echo -e "${GREEN}✅ All packages installed!${NC}"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 4: Launch the app
# ---------------------------------------------------------------------------
echo -e "${GREEN}🚀 Launching EBI Classifier...${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo "   A web browser will open automatically with the app."
echo "   If not, open this URL in your browser:"
echo ""
echo -e "   ${GREEN}http://127.0.0.1:7860${NC}"
echo ""
echo "   To stop the app, close this Terminal window."
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Launch the Gradio app
python app.py

# Keep window open if there was an error
echo ""
echo "App has stopped. Press Enter to close this window..."
read
