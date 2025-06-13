#!/bin/bash
# vLLM 기반 VLM 시스템 설치 스크립트 (Cross-Platform 지원)

echo "🚀 vLLM 기반 VLM 시스템 설치를 시작합니다..."
echo ""
echo "📋 설치 과정:"
echo "   1. 시스템 환경 감지 (OS, 아키텍처, GPU)"
echo "   2. Python 버전 확인"
echo "   3. 가상환경 생성/활성화"
echo "   4. PyTorch 설치 (환경별 최적화)"
echo "   5. vLLM 및 의존성 설치"
echo "   6. 설정 파일 생성"
echo "   7. 설치 검증"
echo ""
echo "⚠️  주의사항:"
echo "   - 가상환경을 사용하여 시스템 Python과 분리됩니다"
echo "   - Apple Silicon Mac의 경우 MPS 가속을 지원합니다"
echo "   - 설치 시간은 환경에 따라 5-30분 소요될 수 있습니다"
echo ""
read -p "계속 진행하시겠습니까? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "❌ 설치가 취소되었습니다."
    exit 0
fi
echo ""

# OS 및 아키텍처 감지
echo "🔍 시스템 환경 감지 중..."
OS=$(uname -s)
ARCH=$(uname -m)
echo "📟 운영체제: $OS"
echo "🏗️  아키텍처: $ARCH"

# 환경 변수 초기화
DEVICE_TYPE="cpu"
PYTORCH_INDEX_URL=""
INSTALL_VLLM=true
MPS_AVAILABLE=false

# OS별 환경 설정
case "$OS" in
    "Darwin")
        echo "🍎 macOS 환경 감지됨"
        if [ "$ARCH" = "arm64" ]; then
            echo "🔥 Apple Silicon (M1/M2/M3) 감지됨"
            DEVICE_TYPE="mps"
            MPS_AVAILABLE=true
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"  # Apple Silicon용 CPU 빌드 사용
        else
            echo "💻 Intel Mac 감지됨"
            DEVICE_TYPE="cpu"
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        fi
        ;;
    "Linux")
        echo "🐧 Linux 환경 감지됨"
        if command -v nvidia-smi &> /dev/null; then
            echo "🚀 NVIDIA GPU 감지됨"
            DEVICE_TYPE="cuda"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        else
            echo "💻 CPU 모드로 설정됨"
            DEVICE_TYPE="cpu"
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        fi
        ;;
    *)
        echo "❓ 알 수 없는 OS: $OS - CPU 모드로 진행합니다"
        DEVICE_TYPE="cpu"
        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        ;;
esac

echo "🎯 감지된 디바이스 타입: $DEVICE_TYPE"

# 시스템 요구사항 확인
echo "📋 시스템 요구사항 확인 중..."

# Python 버전 확인
python_version=$(python3 --version 2>&1 | grep -o "[0-9]\+\.[0-9]\+")
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

# 버전 비교 (3.8 이상인지 확인)
if [ "$python_major" -gt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -ge 8 ]); then
    echo "✅ Python $python_version 확인됨"
else
    echo "❌ Python 3.8 이상이 필요합니다 (현재: $python_version)"
    exit 1
fi

# 가상환경 설정 및 관리
echo "🔧 가상환경 설정 중..."

# 기존 가상환경 확인
if [ -d ".venv" ]; then
    echo "📁 기존 가상환경 발견됨"
    read -p "기존 가상환경을 삭제하고 새로 생성하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  기존 가상환경 제거 중..."
        rm -rf .venv
        echo "✅ 기존 가상환경 제거 완료"
    else
        echo "✅ 기존 가상환경 사용"
    fi
fi

# 가상환경 생성
if [ ! -d ".venv" ]; then
    echo "🆕 새 가상환경 생성 중..."
    if python3 -m venv .venv; then
        echo "✅ 가상환경 생성 완료"
    else
        echo "❌ 가상환경 생성 실패"
        echo "💡 해결 방법:"
        echo "   - macOS: brew install python3-venv 또는 xcode-select --install"
        echo "   - Ubuntu/Debian: sudo apt install python3-venv"
        echo "   - CentOS/RHEL: sudo yum install python3-venv"
        exit 1
    fi
else
    echo "✅ 기존 가상환경 사용"
fi

# 가상환경 활성화 확인
echo "🔌 가상환경 활성화 중..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 가상환경 활성화됨"
    echo "🐍 Python 경로: $(which python)"
    echo "📦 pip 경로: $(which pip)"
elif [ -f ".venv/Scripts/activate" ]; then
    # Windows 호환성 (Git Bash 등)
    source .venv/Scripts/activate
    echo "✅ 가상환경 활성화됨 (Windows)"
else
    echo "❌ 가상환경 활성화 파일을 찾을 수 없습니다"
    exit 1
fi

# 가상환경 내 Python 버전 확인
venv_python_version=$(python --version 2>&1 | grep -o "[0-9]\+\.[0-9]\+")
echo "✅ 가상환경 Python 버전: $venv_python_version"

# pip 업그레이드
echo "📦 pip 업그레이드 중..."
if pip install --upgrade pip; then
    echo "✅ pip 업그레이드 완료"
    echo "📦 pip 버전: $(pip --version)"
else
    echo "⚠️  pip 업그레이드 실패 - 계속 진행합니다"
fi

# 기본 도구 설치
echo "🔧 기본 도구 설치 중..."
pip install wheel setuptools --upgrade

# Apple Silicon MPS 지원 재확인 (가상환경 내에서)
if [ "$MPS_AVAILABLE" = true ]; then
    echo "🔥 Apple Silicon MPS 지원 재확인 중..."
    if python -c "
import sys
import platform
print(f'✅ 가상환경 내 Python: {sys.version}')
print(f'✅ 플랫폼: {platform.platform()}')
print(f'✅ 아키텍처: {platform.machine()}')
print('🍎 Apple Silicon MPS 준비 완료')
" 2>/dev/null; then
        echo "✅ Apple Silicon MPS 환경 준비 완료"
    else
        echo "⚠️  Apple Silicon MPS 환경 확인 실패 - CPU 모드로 전환"
        DEVICE_TYPE="cpu"
        MPS_AVAILABLE=false
    fi
fi

# PyTorch 설치 (환경별 최적화)
echo "🔥 PyTorch 설치 중... (디바이스: $DEVICE_TYPE)"
case "$DEVICE_TYPE" in
    "cuda")
        echo "🚀 CUDA 지원 PyTorch 설치 중..."
        pip install torch torchvision --index-url $PYTORCH_INDEX_URL
        ;;
    "mps")
        echo "🍎 Apple Silicon 최적화 PyTorch 설치 중..."
        pip install torch torchvision --index-url $PYTORCH_INDEX_URL
        # MPS 동작 테스트
        echo "🧪 MPS 동작 테스트 중..."
        if python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'MPS 지원: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    z = torch.matmul(x, y)
    print(f'✅ MPS 연산 테스트 성공: {z.shape}')
    print('🔥 Apple Silicon MPS 가속 사용 가능')
else:
    print('⚠️  MPS 사용 불가 - CPU 모드로 전환')
" 2>/dev/null; then
            echo "✅ MPS 동작 테스트 성공"
        else
            echo "⚠️  MPS 동작 테스트 실패 - CPU 모드로 전환"
            DEVICE_TYPE="cpu"
        fi
        ;;
    "cpu")
        echo "💻 CPU 버전 PyTorch 설치 중..."
        pip install torch torchvision --index-url $PYTORCH_INDEX_URL
        ;;
esac

# 기본 의존성 설치 (PyTorch, vLLM 제외)
echo "📚 기본 의존성 설치 중..."
echo "⚠️  PyTorch와 vLLM은 환경별로 별도 설치됩니다"

# requirements.txt에서 PyTorch, vLLM 관련 줄 제외하고 설치
if [ -f "requirements.txt" ]; then
    # 임시 requirements 파일 생성 (PyTorch, vLLM 제외)
    grep -v -E "^torch|^torchvision|^vllm|^git\+.*transformers" requirements.txt > requirements_temp.txt
    
    echo "📦 설치할 패키지 목록:"
    cat requirements_temp.txt | grep -v "^#" | grep -v "^$" | head -10
    echo "..."
    
    if pip install -r requirements_temp.txt; then
        echo "✅ 기본 의존성 설치 완료"
    else
        echo "⚠️  일부 의존성 설치 실패 - 계속 진행합니다"
    fi
    
    # 임시 파일 정리
    rm -f requirements_temp.txt
else
    echo "⚠️  requirements.txt 파일을 찾을 수 없습니다"
fi

# transformers 별도 설치
echo "🤗 Transformers 라이브러리 설치 중..."
if pip install "transformers>=4.45.0" "accelerate>=0.25.0"; then
    echo "✅ Transformers 설치 완료"
else
    echo "⚠️  Transformers 설치 실패 - 기본 버전으로 재시도"
    pip install transformers accelerate
fi

# vLLM 설치 시도 (환경별 분기)
echo "⚡ vLLM 설치 중..."
if [ "$DEVICE_TYPE" = "mps" ]; then
    echo "🍎 Apple Silicon의 경우 vLLM 호환성 확인 중..."
    if pip install "vllm>=0.7.2" --no-deps; then
        echo "✅ vLLM 기본 설치 성공"
        # Apple Silicon에서 필요한 추가 의존성 설치
        pip install "transformers>=4.45.0" "torch>=2.0.0"
        vllm_installed=true
    else
        echo "⚠️  Apple Silicon에서 vLLM 설치 실패 - transformers fallback 모드로 동작합니다"
        vllm_installed=false
        INSTALL_VLLM=false
    fi
elif pip install "vllm>=0.7.2"; then
    echo "✅ vLLM 설치 성공"
    vllm_installed=true
else
    echo "⚠️  vLLM 설치 실패 - transformers fallback 모드로 동작합니다"
    vllm_installed=false
    INSTALL_VLLM=false
fi

# flash-attention 설치 시도 (환경별 분기)
echo "🚄 Flash Attention 설치 시도 중..."
if [ "$DEVICE_TYPE" = "mps" ]; then
    echo "🍎 Apple Silicon의 경우 Flash Attention 대신 최적화된 attention 사용"
    echo "⚠️  Flash Attention은 Apple Silicon에서 지원되지 않습니다"
elif [ "$DEVICE_TYPE" = "cuda" ]; then
    if pip install flash-attn --no-build-isolation; then
        echo "✅ Flash Attention 설치 성공"
    else
        echo "⚠️  Flash Attention 설치 실패 - 계속 진행합니다"
    fi
else
    echo "💻 CPU 모드에서는 Flash Attention을 건너뜁니다"
fi

# 설정 디렉토리 생성
echo "📁 설정 디렉토리 생성 중..."
mkdir -p configs
mkdir -p models
mkdir -p uploads
mkdir -p logs

# vLLM 설정 생성
if [ "$vllm_installed" = true ]; then
    echo "⚙️  vLLM 설정 생성 중..."
    python src/vllm_config.py
fi

# 권한 설정
echo "🔐 권한 설정 중..."
chmod +x examples_vllm.py
chmod +x test_vllm_integration.py
find src/ -name "*.py" -exec chmod +x {} \;

# 설치 검증 (환경별)
echo "🧪 설치 검증 중..."
echo "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')" | python

if [ "$vllm_installed" = true ]; then
    echo "import vllm; print(f'vLLM: {vllm.__version__}')" | python
fi

echo "from PIL import Image; print('PIL: OK')" | python

# 디바이스별 추가 검증
case "$DEVICE_TYPE" in
    "mps")
        echo "🍎 Apple Silicon MPS 디바이스 검증 중..."
        python3 -c "
import torch
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(10, 10, device=device)
    print('✅ MPS 디바이스 테스트 성공')
else:
    print('⚠️  MPS 디바이스 사용 불가')
" 2>/dev/null || echo "⚠️  MPS 테스트 실패"
        ;;
    "cuda")
        echo "🚀 CUDA 디바이스 검증 중..."
        python3 -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(10, 10, device=device)
    print(f'✅ CUDA 디바이스 테스트 성공 (GPU: {torch.cuda.get_device_name()})')
else:
    print('⚠️  CUDA 디바이스 사용 불가')
" 2>/dev/null || echo "⚠️  CUDA 테스트 실패"
        ;;
    "cpu")
        echo "💻 CPU 디바이스 검증 중..."
        python3 -c "
import torch
x = torch.randn(10, 10)
print('✅ CPU 디바이스 테스트 성공')
" 2>/dev/null || echo "⚠️  CPU 테스트 실패"
        ;;
esac

# 설치 완료 메시지 (환경별 맞춤)
echo ""
echo "🎉 설치가 완료되었습니다!"
echo "🎯 감지된 환경: $OS ($ARCH) - $DEVICE_TYPE 모드"
echo ""
echo "📋 다음 단계:"
echo "1. 가상환경 활성화: source .venv/bin/activate"
echo "2. 예제 실행: python examples_vllm.py"
echo "3. 테스트 실행: python test_vllm_integration.py"
echo ""

# 환경별 안내 메시지
case "$DEVICE_TYPE" in
    "mps")
        echo "🍎 Apple Silicon 최적화 설정:"
        if [ "$vllm_installed" = true ]; then
            echo "✅ vLLM이 설치되어 MPS 가속이 가능합니다"
            echo "💡 Apple Silicon 최적화 팁:"
            echo "   - MPS 디바이스 자동 사용됨"
            echo "   - 메모리 효율성을 위해 작은 배치 크기 권장"
            echo "   - Unified Memory 활용으로 큰 모델 로드 가능"
        else
            echo "⚠️  vLLM 설치 실패 - transformers + MPS fallback 모드 사용"
            echo "💡 Apple Silicon에서 transformers 사용 시:"
            echo "   - torch.device('mps') 자동 사용"
            echo "   - Flash Attention 대신 기본 attention 사용"
        fi
        ;;
    "cuda")
        echo "🚀 NVIDIA GPU 최적화 설정:"
        if [ "$vllm_installed" = true ]; then
            echo "✅ vLLM이 설치되어 고성능 CUDA 추론이 가능합니다"
            echo "📊 성능 설정:"
            echo "   - configs/vllm_auto.json: 자동 감지 설정"
            echo "   - configs/vllm_throughput.json: 처리량 최적화"
            echo "   - configs/vllm_latency.json: 지연시간 최적화"
        else
            echo "⚠️  vLLM 설치 실패 - transformers + CUDA fallback 모드 사용"
        fi
        ;;
    "cpu")
        echo "� CPU 모드 최적화 설정:"
        if [ "$vllm_installed" = true ]; then
            echo "✅ vLLM CPU 모드가 설치되었습니다"
        else
            echo "⚠️  vLLM 설치 실패 - transformers CPU 모드 사용"
        fi
        echo "💡 CPU 최적화 팁:"
        echo "   - OpenMP 스레드 수 조정: export OMP_NUM_THREADS=4"
        echo "   - 작은 모델 사용 권장 (7B 이하)"
        ;;
esac

echo ""
echo "🆘 환경별 문제 해결:"
case "$DEVICE_TYPE" in
    "mps")
        echo "   - MPS 오류: Python 3.8+ 및 macOS 12.3+ 필요"
        echo "   - 메모리 부족: 더 작은 모델 사용 또는 배치 크기 감소"
        echo "   - 모델 호환성: Hugging Face transformers 공식 지원 모델 사용"
        ;;
    "cuda")
        echo "   - CUDA 오류: nvidia-smi 확인 후 PyTorch 재설치"
        echo "   - 메모리 부족: gpu_memory_utilization 값 조정"
        echo "   - vLLM 오류: CUDA 11.8+ 및 적절한 GPU 메모리 필요"
        ;;
    "cpu")
        echo "   - 성능 느림: 작은 모델 사용 및 스레드 수 조정"
        echo "   - 메모리 부족: swap 메모리 확인 및 모델 크기 조정"
        ;;
esac
echo "   - 모델 로드 실패: models/ 폴더에 모델 다운로드 필요"
echo ""
echo "� 가상환경 사용법:"
echo "   - 활성화: source .venv/bin/activate"
echo "   - 비활성화: deactivate"
echo "   - 상태 확인: which python"
echo "   - 패키지 목록: pip list"
echo ""
echo "💡 중요한 팁:"
echo "   - 터미널을 새로 열 때마다 'source .venv/bin/activate' 실행 필요"
echo "   - IDE에서 Python 인터프리터를 .venv/bin/python으로 설정"
echo "   - 가상환경이 활성화되면 프롬프트에 (.venv) 표시됨"
echo ""
echo "�📖 사용법: README.md 참조"
