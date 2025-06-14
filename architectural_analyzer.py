#!/usr/bin/env python3
"""
🏗️ 건축 도면 분석 시스템 (통합 버전)
PDF 건축 도면을 분석하여 상세한 정보를 추출합니다.

사용법:
    python architectural_analyzer.py [PDF파일경로] [옵션]
    
옵션:
    --output-dir: 출력 디렉토리 (기본값: analysis_results)
    --format: 출력 형식 (json, html, both) (기본값: both)
    --model: 사용할 모델 (기본값: .env에서 읽음)
    --backend: 백엔드 (vllm, transformers) (기본값: transformers)
    --verbose: 상세 로그 출력
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

# 필수 라이브러리 import
try:
    from PIL import Image
    import fitz  # PyMuPDF
    from dotenv import load_dotenv
    
    # 프로젝트 모듈들
    from env_config import get_env_config
    from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
    from vllm_config import VLLMConfig
    
    print("✅ 모든 필수 라이브러리 로드 완료")
except ImportError as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    print("pip install -r requirements.txt 를 실행해주세요.")
    sys.exit(1)

# 로깅 설정
def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('architectural_analysis.log')
        ]
    )
    return logging.getLogger(__name__)

# PDF 처리 클래스
class PDFProcessor:
    """PDF 처리 및 이미지 변환"""
    
    def __init__(self, temp_dir: str = "temp_analysis"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def convert_pdf_to_images(self, pdf_path: str, max_pages: int = 10) -> List[tuple]:
        """PDF를 이미지로 변환 (최적화된 버전)"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
        print(f"📄 PDF 변환 시작: {pdf_path.name}")
        
        doc = fitz.open(pdf_path)
        images = []
        
        # 페이지 수 제한
        total_pages = min(len(doc), max_pages)
        
        for page_num in range(total_pages):
            try:
                page = doc.load_page(page_num)
                
                # 고품질 렌더링 (200 DPI)
                mat = fitz.Matrix(200/72, 200/72)
                pix = page.get_pixmap(matrix=mat)
                
                # PIL Image로 변환
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                
                # 이미지 크기 최적화 (최대 1024x1024, 품질 유지)
                if img.width > 1024 or img.height > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # 임시 파일로 저장
                img_path = self.temp_dir / f"page_{page_num + 1}.png"
                img.save(img_path, "PNG", optimize=True)
                
                images.append((img, str(img_path), page_num + 1))
                print(f"   ✅ 페이지 {page_num + 1} 변환 완료 ({img.size[0]}x{img.size[1]})")
                
            except Exception as e:
                print(f"   ❌ 페이지 {page_num + 1} 변환 실패: {e}")
                continue
        
        doc.close()
        print(f"📄 총 {len(images)}개 페이지 변환 완료")
        return images
    
    def cleanup(self):
        """임시 파일 정리"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("🧹 임시 파일 정리 완료")

# 분석 결과 처리 클래스
class AnalysisResultProcessor:
    """분석 결과 처리 및 출력"""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_results(self, results: Dict[str, Any], format_type: str = "both") -> Dict[str, str]:
        """분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        if format_type in ["json", "both"]:
            json_file = self.output_dir / f"analysis_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            output_files["json"] = str(json_file)
            print(f"💾 JSON 결과 저장: {json_file}")
        
        if format_type in ["html", "both"]:
            html_file = self.output_dir / f"analysis_{timestamp}.html"
            html_content = self._generate_html_report(results)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            output_files["html"] = str(html_file)
            print(f"🌐 HTML 리포트 저장: {html_file}")
        
        return output_files
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """HTML 리포트 생성"""
        html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>건축 도면 분석 리포트</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #667eea; background: #f8f9fa; }
        .page-analysis { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
        .analysis-type { font-weight: bold; color: #667eea; margin-bottom: 10px; }
        .result-text { background: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏗️ 건축 도면 분석 리포트</h1>
        <p class="timestamp">생성 시간: {timestamp}</p>
    </div>

    <div class="section">
        <h2>📊 분석 개요</h2>
        <p><strong>분석된 페이지 수:</strong> {page_count}</p>
        <p><strong>PDF 파일:</strong> {pdf_file}</p>
    </div>

    {page_analyses}

    <div class="section">
        <h2>📋 전체 요약</h2>
        <div class="result-text">
            <p>이 리포트는 AI 기반 건축 도면 분석 시스템으로 생성되었습니다.</p>
            <p>더 정확한 분석을 위해서는 전문가의 검토가 필요할 수 있습니다.</p>
        </div>
    </div>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
        <p>Generated by Architectural VLM Analysis System</p>
    </footer>
</body>
</html>
        """
        
        # 페이지별 분석 내용 생성
        page_analyses = ""
        page_count = 0
        
        for page_key, page_data in results.items():
            if page_key.startswith("page_"):
                page_count += 1
                page_num = page_key.split("_")[1]
                
                page_html = f"""
                <div class="page-analysis">
                    <h3>📄 페이지 {page_num} 분석</h3>
                """
                
                for analysis_type, analysis_result in page_data.items():
                    page_html += f"""
                    <div class="analysis-type">{analysis_type}</div>
                    <div class="result-text">
                        <pre>{analysis_result}</pre>
                    </div>
                    """
                
                page_html += "</div>"
                page_analyses += page_html
        
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            page_count=page_count,
            pdf_file=results.get("metadata", {}).get("source_file", "Unknown"),
            page_analyses=page_analyses
        )

# 메인 분석기 클래스
class ArchitecturalAnalyzer:
    """통합 건축 도면 분석기"""
    
    def __init__(self, model_name: str = None, backend: str = "transformers", verbose: bool = False):
        self.logger = setup_logging(verbose)
        self.model_name = model_name
        self.backend = backend
        
        # 환경 설정 로드
        load_dotenv()
        self.config = get_env_config()
        
        if not self.model_name:
            self.model_name = self.config.get("MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
        
        # VLM 분석기 초기화
        self.vlm_analyzer = None
        self._initialize_analyzer()
        
        # 처리기들 초기화
        self.pdf_processor = PDFProcessor()
        self.result_processor = AnalysisResultProcessor()
    
    def _initialize_analyzer(self):
        """VLM 분석기 초기화"""
        try:
            print(f"🤖 {self.model_name} 모델 로딩 중...")
            use_vllm = (self.backend == "vllm")
            self.vlm_analyzer = QwenVLMAnalyzer(
                model_path=self.model_name,
                use_vllm=use_vllm
            )
            print("✅ 모델 로딩 완료")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def analyze_pdf(self, pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
        """PDF 건축 도면 전체 분석"""
        print(f"🚀 건축 도면 분석 시작: {pdf_path}")
        
        results = {
            "metadata": {
                "source_file": pdf_path,
                "analysis_time": datetime.now().isoformat(),
                "model_used": self.model_name,
                "backend": self.backend
            }
        }
        
        try:
            # 1. PDF를 이미지로 변환
            images = self.pdf_processor.convert_pdf_to_images(pdf_path, max_pages)
            
            # 2. 각 페이지 분석
            for img, img_path, page_num in images:
                print(f"\n📋 페이지 {page_num} 분석 중...")
                page_results = self._analyze_single_page(img, page_num)
                results[f"page_{page_num}"] = page_results
            
            # 3. 전체 요약
            results["summary"] = self._generate_summary(results)
            
            print("\n✅ 전체 분석 완료!")
            return results
            
        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {e}")
            raise
        finally:
            # 정리
            self.pdf_processor.cleanup()
            if self.vlm_analyzer:
                self.vlm_analyzer.cleanup()
    
    def _analyze_single_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """단일 페이지 분석"""
        page_results = {}
        
        # 분석 프롬프트들
        analysis_prompts = {
            "기본_구조_분석": """
            이 건축 평면도를 분석하여 다음 내용을 한국어로 설명해주세요:
            
            1. 전체 건물 유형 (아파트, 주택, 상가 등)
            2. 주요 공간들 (방, 거실, 주방, 화장실 등)
            3. 문과 창문의 위치
            4. 벽체의 구조
            5. 특별한 건축적 특징
            
            명확하고 상세하게 분석해주세요.
            """,
            
            "공간_분석": """
            이 평면도의 공간 구성을 분석하여 JSON 형식으로 제공해주세요:
            
            {
              "rooms": [
                {"name": "방이름", "type": "용도", "position": "위치", "features": ["특징1", "특징2"]}
              ],
              "circulation": "동선 분석",
              "total_layout": "전체 레이아웃 특징"
            }
            """,
            
            "치수_및_스케일": """
            이 도면에서 발견할 수 있는 치수 정보와 스케일을 분석해주세요:
            
            1. 표시된 치수들
            2. 도면의 스케일
            3. 추정 면적
            4. 주요 치수 (방 크기, 문 폭 등)
            
            구체적인 수치와 함께 설명해주세요.
            """,
            
            "설비_및_시설": """
            이 평면도에서 설비와 시설물을 찾아 분석해주세요:
            
            1. 전기 설비 (콘센트, 스위치, 조명 등)
            2. 급수/배수 설비
            3. 주방 시설
            4. 화장실 설비
            5. 기타 특수 시설
            
            각 시설의 위치와 특징을 설명해주세요.
            """
        }
        
        # 각 프롬프트로 분석 실행
        for analysis_type, prompt in analysis_prompts.items():
            try:
                print(f"   🔍 {analysis_type} 진행 중...")
                result = self.vlm_analyzer.analyze_image(image, prompt)
                page_results[analysis_type] = result
                print(f"   ✅ {analysis_type} 완료")
            except Exception as e:
                print(f"   ❌ {analysis_type} 실패: {e}")
                page_results[analysis_type] = f"분석 실패: {str(e)}"
        
        return page_results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """전체 분석 결과 요약"""
        summary = {
            "total_pages": len([k for k in results.keys() if k.startswith("page_")]),
            "analysis_types_completed": [],
            "key_findings": [],
            "recommendations": []
        }
        
        # 분석 유형 수집
        for page_key, page_data in results.items():
            if page_key.startswith("page_"):
                for analysis_type in page_data.keys():
                    if analysis_type not in summary["analysis_types_completed"]:
                        summary["analysis_types_completed"].append(analysis_type)
        
        # 주요 발견사항 (간단한 예시)
        summary["key_findings"] = [
            "건축 도면 분석이 성공적으로 완료되었습니다.",
            f"총 {summary['total_pages']}개 페이지가 분석되었습니다.",
            f"{len(summary['analysis_types_completed'])}가지 분석 유형이 적용되었습니다."
        ]
        
        summary["recommendations"] = [
            "AI 분석 결과는 참고용으로 사용하시고, 정확한 검토를 위해 전문가 상담을 받으시기 바랍니다.",
            "도면의 스케일과 치수 정보를 다시 한번 확인해주세요.",
            "설비 계획에 대해서는 관련 전문가와 상의하시기 바랍니다."
        ]
        
        return summary

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="🏗️ 건축 도면 분석 시스템")
    parser.add_argument("pdf_path", nargs="?", help="분석할 PDF 파일 경로")
    parser.add_argument("--output-dir", default="analysis_results", help="출력 디렉토리")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both", help="출력 형식")
    parser.add_argument("--model", help="사용할 모델명")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="transformers", help="백엔드")
    parser.add_argument("--max-pages", type=int, default=5, help="분석할 최대 페이지 수")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # PDF 파일 경로 결정
    pdf_path = args.pdf_path
    if not pdf_path:
        # 기본 파일 찾기
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            pdf_files = list(uploads_dir.glob("*.pdf"))
            if pdf_files:
                pdf_path = str(pdf_files[0])
                print(f"📁 자동 선택된 파일: {pdf_path}")
            else:
                print("❌ uploads 디렉토리에 PDF 파일이 없습니다.")
                return
        else:
            print("❌ PDF 파일을 지정해주세요.")
            print("사용법: python architectural_analyzer.py [PDF파일경로]")
            return
    
    if not Path(pdf_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    try:
        # 분석기 초기화
        analyzer = ArchitecturalAnalyzer(
            model_name=args.model,
            backend=args.backend,
            verbose=args.verbose
        )
        
        # 결과 처리기 설정
        analyzer.result_processor = AnalysisResultProcessor(args.output_dir)
        
        # 분석 실행
        results = analyzer.analyze_pdf(pdf_path, args.max_pages)
        
        # 결과 저장
        output_files = analyzer.result_processor.save_results(results, args.format)
        
        print("\n🎉 분석 완료!")
        print("📂 저장된 파일들:")
        for file_type, file_path in output_files.items():
            print(f"   {file_type.upper()}: {file_path}")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # 필요한 import들 추가
    from io import BytesIO
    
    exit(main())
