#!/usr/bin/env python3
"""
uploads 폴더의 건축 도면 분석 스크립트
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

def find_files_in_uploads():
    """uploads 폴더에서 파일 찾기"""
    uploads_dir = Path("uploads")
    
    if not uploads_dir.exists():
        uploads_dir.mkdir()
        print("📁 Created uploads/ directory")
        return [], []
    
    # PDF와 이미지 파일 찾기
    pdf_files = list(uploads_dir.glob("*.pdf"))
    image_files = (
        list(uploads_dir.glob("*.png")) + 
        list(uploads_dir.glob("*.jpg")) + 
        list(uploads_dir.glob("*.jpeg")) +
        list(uploads_dir.glob("*.bmp")) +
        list(uploads_dir.glob("*.tiff"))
    )
    
    return pdf_files, image_files

def analyze_image_file(image_path, analyzer):
    """이미지 파일 분석"""
    print(f"🖼️  Analyzing: {image_path.name}")
    
    try:
        # 이미지 로드
        image = Image.open(image_path)
        print(f"   📏 Original size: {image.size[0]}x{image.size[1]}")
        
        # 이미지 크기 제한 (메모리 절약)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"   📏 Resized to: {image.size[0]}x{image.size[1]}")
        
        # 건축 도면 분석
        print("   🧠 Analyzing architectural elements...")
        
        # 간단한 분석 프롬프트
        prompt = """

{
  "improved_prompt": {
    "system": [
      "당신은 건축·토목·설비 전 도면을 이해하는 AI 검토 엔지니어입니다.",
      "모든 답변은 반드시 **한국어**로 작성하며, 하나의 JSON 오브젝트만 출력합니다(텍스트·마크다운 사용 금지).",
      "JSON 키 구조는 { \"요약\", \"세부\", \"액션아이템\", \"근거\" } 네 가지로 고정합니다.",
      "각 근거 항목은 { \"sheet\": <pageNo>, \"bbox\": [x1, y1, x2, y2] } 형식의 배열로 제시합니다.",
      "근거에 없는 정보는 문자열 \"근거 부족\"으로 명시하고 절대 추측하지 않습니다."
    ],
    "format_rules": [
      "출력 예시:",
      "{",
      "  \"요약\": \"...\",",
      "  \"세부\": \"...\",",
      "  \"액션아이템\": \"...\",",
      "  \"근거\": [",
      "    { \"sheet\": 12, \"bbox\": [100, 200, 350, 480] }",
      "  ]",
      "}",
      "위 예시와 동일한 키·순서를 유지합니다.",
      "도면 Deep-Link는 필요 시 값 내부에 `sheet=<pageNo>&bbox=<x1,y1,x2,y2>` 문자열로 삽입해도 무방합니다."
    ],
    "tool_instructions": [
      "멀티모달 모델(Qwen-VL, LLaVA 등)은 `BLIP_VISION_Q` 토큰으로 이미지 임베딩을 호출합니다.",
      "LangGraph 상태 `Validate` 단계에서 Self-Critique 수행 후 score<0.5이면 Retrieval을 1회 재시도합니다."
    ],
    "placeholders": [
      "<context>",
      "<images>",
      "<user_question>"
    ],
    "assistant_placeholder": "<assistant_response_JSON>"
  },
  "rationale": {
    "목표_반영": "사용자 요청인 ‘JSON 파일로만 대답’ 요구사항을 충족하도록 JSON 출력 강제 규칙 추가",
    "일관성": "기존 ‘요약 ▶ 세부 ▶ 액션아이템’ 3단계 구조를 동일한 키로 고정해 혼동 방지",
    "근거명시": "근거를 배열로 분리해 시트번호·bbox를 구조화, Deep-Link 호환성 유지",
    "검증강화": "Self-Critique 후 재시도 규칙을 명문화해 품질 확보"
  }
}


        """
        
        result = analyzer.analyze_image(image, prompt)
        
        if result:
            print("   ✅ Analysis completed")
            return {
                "file": image_path.name,
                "size": f"{image.size[0]}x{image.size[1]}",
                "analysis": result
            }
        else:
            print("   ❌ Analysis failed")
            return {"file": image_path.name, "error": "Analysis failed"}
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {"file": image_path.name, "error": str(e)}

def analyze_pdf_file(pdf_path, analyzer):
    """PDF 파일 분석"""
    print(f"📄 Analyzing PDF: {pdf_path.name}")
    
    try:
        # PDF 열기
        doc = fitz.open(pdf_path)
        results = []
        
        print(f"   📋 Found {len(doc)} pages")
        
        for page_num in range(min(len(doc), 3)):  # 최대 3페이지만 처리
            print(f"   📄 Processing page {page_num + 1}...")
            
            page = doc.load_page(page_num)
            
            # 이미지로 변환 (해상도 조정)
            mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            
            # PIL Image로 변환
            import io
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            
            # 크기 제한
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            print(f"      📏 Page size: {image.size[0]}x{image.size[1]}")
            
            # 분석
            prompt = f"""
            이것은 PDF 문서의 {page_num + 1}페이지입니다. 이 건축 도면을 분석해주세요:
            
            1. 도면 유형 (평면도, 입면도, 단면도, 상세도 등)
            2. 주요 건축 요소들
            3. 공간 구성
            4. 특이사항이나 중요한 정보
            
            간결하고 명확하게 답변해주세요.
            """
            
            result = analyzer.analyze_image(image, prompt)
            
            if result:
                results.append({
                    "page": page_num + 1,
                    "size": f"{image.size[0]}x{image.size[1]}",
                    "analysis": result
                })
                print(f"      ✅ Page {page_num + 1} analyzed")
            else:
                results.append({
                    "page": page_num + 1,
                    "error": "Analysis failed"
                })
                print(f"      ❌ Page {page_num + 1} failed")
        
        doc.close()
        
        return {
            "file": pdf_path.name,
            "total_pages": len(doc),
            "processed_pages": len(results),
            "pages": results
        }
        
    except Exception as e:
        print(f"   ❌ Error processing PDF: {e}")
        return {"file": pdf_path.name, "error": str(e)}

def main():
    print("🏗️  Uploads Folder Analysis")
    print("=" * 40)
    
    # uploads 폴더에서 파일 찾기
    pdf_files, image_files = find_files_in_uploads()
    
    if not pdf_files and not image_files:
        print("❌ No files found in uploads/ folder")
        print("\n📝 Instructions:")
        print("   1. Place your architectural drawings in the uploads/ folder")
        print("   2. Supported formats: PDF, PNG, JPG, JPEG, BMP, TIFF")
        print("   3. Run this script again")
        return
    
    print(f"📁 Found files:")
    print(f"   📄 PDF files: {len(pdf_files)}")
    print(f"   🖼️  Image files: {len(image_files)}")
    
    for pdf in pdf_files:
        print(f"      📄 {pdf.name}")
    for img in image_files:
        print(f"      🖼️  {img.name}")
    
    # VLM 분석기 로드
    try:
        from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
        
        print("\n🧠 Loading VLM analyzer...")
        analyzer = QwenVLMAnalyzer(use_vllm=False)
        
        if not analyzer.load_model():
            print("❌ Failed to load VLM model")
            return
        
        print("✅ VLM analyzer loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading analyzer: {e}")
        return
    
    # 분석 결과 저장
    results = {
        "pdf_files": [],
        "image_files": [],
        "summary": {}
    }
    
    # PDF 파일들 분석
    for pdf_file in pdf_files:
        result = analyze_pdf_file(pdf_file, analyzer)
        results["pdf_files"].append(result)
    
    # 이미지 파일들 분석
    for image_file in image_files:
        result = analyze_image_file(image_file, analyzer)
        results["image_files"].append(result)
    
    # 요약 정보
    results["summary"] = {
        "total_pdf_files": len(pdf_files),
        "total_image_files": len(image_files),
        "total_files": len(pdf_files) + len(image_files),
        "successful_analyses": 0,
        "failed_analyses": 0
    }
    
    # 성공/실패 카운트
    for pdf_result in results["pdf_files"]:
        if "error" not in pdf_result:
            results["summary"]["successful_analyses"] += 1
        else:
            results["summary"]["failed_analyses"] += 1
    
    for img_result in results["image_files"]:
        if "error" not in img_result:
            results["summary"]["successful_analyses"] += 1
        else:
            results["summary"]["failed_analyses"] += 1
    
    # 결과 저장
    output_file = "uploads_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # 요약 출력
    print(f"\n📊 SUMMARY:")
    print(f"   📋 Total files: {results['summary']['total_files']}")
    print(f"   ✅ Successful: {results['summary']['successful_analyses']}")
    print(f"   ❌ Failed: {results['summary']['failed_analyses']}")
    
    # 클린업
    analyzer.cleanup()
    print("\n🎉 Analysis completed!")

if __name__ == "__main__":
    main()
