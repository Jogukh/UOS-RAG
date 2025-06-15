#!/bin/bash

# VLM Environment Setup Script
# Optimized for different platforms including Apple Silicon

set -e  # Exit on any error

echo "🚀 VLM Environment Setup Starting..."
echo "================================================"

# Function to detect system architecture and OS
detect_system() {
    local arch=$(uname -m)
    local os=$(uname -s)
    
    echo "Detected System: $os"
    echo "Architecture: $arch"
    
    # Check for Apple Silicon
    if [[ "$os" == "Darwin" && ("$arch" == "arm64" || "$arch" == "aarch64") ]]; then
        echo "✅ Apple Silicon Mac detected"
        return 0  # Apple Silicon
    elif [[ "$os" == "Darwin" ]]; then
        echo "✅ Intel Mac detected"
        return 1  # Intel Mac
    elif [[ "$os" == "Linux" ]]; then
        echo "✅ Linux system detected"
        return 2  # Linux
    else
        echo "⚠️  Unknown system detected"
        return 3  # Unknown
    fi
}

# Function to check Python version
check_python() {
    echo "🐍 Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo "Python version: $python_version"
        
        # Check if version is 3.8 or higher
        local major_minor=$(echo $python_version | cut -d'.' -f1,2)
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            echo "✅ Python version is compatible"
            return 0
        else
            echo "❌ Python 3.8+ required, found $python_version"
            return 1
        fi
    else
        echo "❌ Python3 not found"
        return 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    echo "📦 Setting up virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        echo "✅ Virtual environment created"
    else
        echo "✅ Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    echo "✅ Pip upgraded"
}

# Function to install base dependencies
install_base_deps() {
    echo "📚 Installing base dependencies..."
    
    # Install core ML packages
    pip install torch torchvision torchaudio
    pip install transformers
    pip install accelerate
    pip install pillow
    pip install numpy
    pip install requests
    pip install opencv-python
    
    echo "✅ Base dependencies installed"
}

# Function to install Apple Silicon optimized packages
install_apple_silicon_deps() {
    echo "🍎 Installing Apple Silicon optimized packages..."
    
    # Install MPS optimized torch if not already installed
    pip install --upgrade torch torchvision torchaudio
    
    # Install additional packages for Apple Silicon
    pip install psutil
    pip install tqdm
    
    echo "✅ Apple Silicon optimizations installed"
    echo "⚠️  Note: vLLM is not supported on Apple Silicon"
    echo "    Using MPS (Metal Performance Shaders) for GPU acceleration"
}

# Function to install Linux/Intel dependencies
install_linux_intel_deps() {
    echo "🐧 Installing Linux/Intel dependencies..."
    
    # Try to install vLLM for Linux systems
    if pip install vllm; then
        echo "✅ vLLM installed successfully"
    else
        echo "⚠️  vLLM installation failed, continuing with CPU-only setup"
    fi
    
    # Install additional packages
    pip install psutil
    pip install tqdm
    
    echo "✅ Linux/Intel dependencies installed"
}

# Function to install project requirements
install_project_requirements() {
    echo "📋 Installing project requirements..."
    
    if [[ -f "requirements.txt" ]]; then
        # Filter out problematic packages for Apple Silicon
        if detect_system; then  # Apple Silicon
            echo "Filtering requirements for Apple Silicon..."
            grep -v "^vllm" requirements.txt > temp_requirements.txt || true
            if [[ -s temp_requirements.txt ]]; then
                pip install -r temp_requirements.txt
            fi
            rm -f temp_requirements.txt
        else
            pip install -r requirements.txt
        fi
        echo "✅ Project requirements installed"
    else
        echo "⚠️  requirements.txt not found, skipping"
    fi
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')

# Check device availability
if torch.cuda.is_available():
    print('✅ CUDA available')
    print(f'   CUDA devices: {torch.cuda.device_count()}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon GPU) available')
else:
    print('✅ CPU mode available')

# Try importing transformers
try:
    import transformers
    print(f'✅ Transformers version: {transformers.__version__}')
except ImportError:
    print('❌ Transformers import failed')

# Try importing vLLM (will fail on Apple Silicon)
try:
    import vllm
    print(f'✅ vLLM version: {vllm.__version__}')
except ImportError:
    print('⚠️  vLLM not available (expected on Apple Silicon)')
"
}

# Main installation process
main() {
    echo "🔧 Starting VLM environment setup..."
    echo "================================================"
    
    # Detect system
    detect_system
    local system_type=$?
    
    # Check Python
    if ! check_python; then
        echo "❌ Python check failed. Please install Python 3.8+"
        if [[ $(uname -s) == "Darwin" ]]; then
            echo "💡 Install Python via Homebrew: brew install python@3.11"
        fi
        exit 1
    fi
    
    # Setup virtual environment
    setup_venv
    
    # Install base dependencies
    install_base_deps
    
    # Install system-specific dependencies
    case $system_type in
        0)  # Apple Silicon
            install_apple_silicon_deps
            ;;
        1)  # Intel Mac
            install_linux_intel_deps
            ;;
        2)  # Linux
            install_linux_intel_deps
            ;;
        *)  # Unknown
            echo "⚠️  Unknown system, installing base dependencies only"
            ;;
    esac
    
    # Install project requirements
    install_project_requirements
    
    # Verify installation
    verify_installation
    
    echo "================================================"
    echo "🎉 VLM Environment setup completed!"
    echo ""
    echo "💡 Usage Instructions:"
    echo "   1. Activate virtual environment: source venv/bin/activate"
    echo "   2. Run tests: python simple_test.py"
    echo "   3. For Apple Silicon: MPS acceleration will be used automatically"
    echo "   4. For Linux/Intel: CUDA will be used if available"
    echo ""
    echo "📝 Next Steps:"
    echo "   - Test the installation with: python simple_test.py"
    echo "   - Check available models in the models/ directory"
    echo "   - Review logs/ directory for any issues"
    echo "================================================"
}

# Run main function
main "$@"
