#!/usr/bin/env python3
"""
실제 건축 도면 분석 스크립트
"""

import os
import sys
import json
import io
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

def pdf_to_images(pdf_path: str, output_dir: str = "temp_images") -> list:
    """PDF를 이미지로 변환 (메모리 최적화)"""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"📄 Converting PDF to images: {pdf_path.name}")
    
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 메모리 절약을 위해 해상도 조정 (DPI 150)
        mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
        pix = page.get_pixmap(matrix=mat)
        
        # PIL Image로 변환
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        # 이미지 크기 제한 (최대 1024x1024)
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"   📏 Resized to: {img.size[0]}x{img.size[1]}")
        
        # 이미지 저장
        img_path = output_dir / f"page_{page_num + 1}.png"
        img.save(img_path)
        images.append(img)
        
        print(f"   ✅ Page {page_num + 1} converted ({img.size[0]}x{img.size[1]})")
    
    doc.close()
    return images

def analyze_architectural_drawing(image_path_or_pil, analyzer):
    """건축 도면 분석"""
    print("🏗️  Analyzing architectural drawing...")
    
    # 여러 분석 모드로 분석
    analysis_results = {}
    
    # 1. 기본 건축 요소 분석
    print("   🔍 Basic architectural analysis...")
    result1 = analyzer.analyze_image(image_path_or_pil, "architectural_basic")
    analysis_results["architectural_basic"] = result1
    
    # 2. 요소 검출
    print("   🔍 Element detection...")
    result2 = analyzer.analyze_image(image_path_or_pil, "element_detection")
    analysis_results["element_detection"] = result2
    
    # 3. 공간 분석
    print("   🔍 Space analysis...")
    space_prompt = """
    이 건축 평면도를 분석하여 다음 정보를 JSON 형식으로 제공해주세요:
    
    {
      "rooms": [
        {"name": "방 이름", "type": "거실|침실|주방|화장실|현관", "estimated_area": "면적 추정", "position": "위치"}
      ],
      "dimensions": [
        {"element": "요소명", "measurement": "치수", "unit": "단위"}
      ],
      "architectural_features": [
        {"feature": "특징", "description": "설명", "location": "위치"}
      ],
      "overall_layout": "전체 레이아웃 설명"
    }
    """
    result3 = analyzer.analyze_image(image_path_or_pil, space_prompt)
    analysis_results["space_analysis"] = result3
    
    return analysis_results

def main():
    print("🏗️  Architectural Drawing Analysis")
    print("=" * 50)
    
    # PDF 파일 경로
    pdf_path = "uploads/A04-001 단위세대평면도 5.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # PyMuPDF import 체크
        import io
        
        # PDF를 이미지로 변환
        images = pdf_to_images(pdf_path)
        
        if not images:
            print("❌ No images extracted from PDF")
            return
        
        # VLM 분석기 로드
        from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
        
        print("\n🧠 Loading VLM analyzer...")
        analyzer = QwenVLMAnalyzer(use_vllm=False)
        
        if not analyzer.load_model():
            print("❌ Failed to load VLM model")
            return
        
        print("✅ VLM model loaded successfully")
        
        # 각 페이지 분석
        all_results = {}
        
        for i, image in enumerate(images):
            print(f"\n📋 Analyzing page {i + 1}...")
            
            try:
                results = analyze_architectural_drawing(image, analyzer)
                all_results[f"page_{i + 1}"] = results
                
                # 결과 요약 출력
                print(f"   ✅ Page {i + 1} analysis completed")
                
                # 간단한 요약 출력
                if "architectural_basic" in results and results["architectural_basic"]:
                    basic_result = results["architectural_basic"]
                    if "parsed_result" in basic_result:
                        elements = basic_result["parsed_result"].get("detected_elements", [])
                        print(f"   📊 Detected {len(elements)} architectural elements")
                
            except Exception as e:
                print(f"   ❌ Error analyzing page {i + 1}: {e}")
                all_results[f"page_{i + 1}"] = {"error": str(e)}
        
        # 결과 저장
        output_file = "architectural_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # 주요 결과 요약
        print("\n📊 ANALYSIS SUMMARY")
        print("=" * 30)
        
        total_elements = 0
        total_rooms = 0
        
        for page_key, page_results in all_results.items():
            if "error" not in page_results:
                print(f"\n{page_key.upper()}:")
                
                # 건축 요소 수
                if "element_detection" in page_results:
                    det_result = page_results["element_detection"]
                    if det_result and "parsed_result" in det_result:
                        elements = det_result["parsed_result"].get("detected_elements", [])
                        print(f"  🏗️  Architectural elements: {len(elements)}")
                        total_elements += len(elements)
                
                # 공간 분석
                if "space_analysis" in page_results:
                    space_result = page_results["space_analysis"]
                    if space_result and "raw_response" in space_result:
                        print(f"  🏠 Space analysis completed")
                        # JSON 파싱 시도
                        try:
                            import re
                            json_match = re.search(r'\{.*\}', space_result["raw_response"], re.DOTALL)
                            if json_match:
                                space_data = json.loads(json_match.group())
                                if "rooms" in space_data:
                                    print(f"  🚪 Rooms identified: {len(space_data['rooms'])}")
                                    total_rooms += len(space_data['rooms'])
                        except:
                            pass
        
        print(f"\n🎯 TOTAL SUMMARY:")
        print(f"   📋 Pages analyzed: {len([k for k in all_results.keys() if 'error' not in all_results[k]])}")
        print(f"   🏗️  Total elements detected: {total_elements}")
        print(f"   🏠 Total rooms identified: {total_rooms}")
        
        analyzer.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
