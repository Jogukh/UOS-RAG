#!/usr/bin/env python3
"""
uploads 폴더의 건축 도면 분석 스크립트 (텍스트 추출 중심)
"""

import os
import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

def find_files_in_uploads():
    """uploads 폴더 및 하위 프로젝트 폴더에서 파일을 찾습니다."""
    uploads_dir = Path("uploads")
    projects_files = {}

    if not uploads_dir.exists():
        uploads_dir.mkdir(parents=True, exist_ok=True)
        print("📁 Created uploads/ directory")
        return projects_files

    # uploads 폴더 직속 파일 처리 (기본 프로젝트로 간주)
    default_project_name = "_default_project"
    default_pdf_files = list(uploads_dir.glob("*.pdf"))
    # 이미지 파일은 더 이상 처리하지 않음
    if default_pdf_files:
        projects_files[default_project_name] = {
            "path": str(uploads_dir.resolve()),
            "pdf_files": default_pdf_files,
            "image_files": [] # 이미지 파일 목록은 비워둠
        }

    # 하위 폴더 (프로젝트) 탐색
    for item in uploads_dir.iterdir():
        if item.is_dir():
            project_name = item.name
            project_path = item
            
            pdf_files = list(project_path.glob("**/*.pdf")) # 하위 폴더의 PDF 까지 검색
            # 이미지 파일은 더 이상 처리하지 않음
            
            if pdf_files:
                if project_name in projects_files:
                    projects_files[project_name]["pdf_files"].extend(pdf_files)
                else:
                    projects_files[project_name] = {
                        "path": str(project_path.resolve()),
                        "pdf_files": pdf_files,
                        "image_files": [] # 이미지 파일 목록은 비워둠
                    }
    
    return projects_files

def extract_text_from_pdf_page(page):
    """PDF 페이지에서 텍스트를 추출합니다."""
    try:
        text = page.get_text("text")
        if not text.strip():  # 텍스트가 비어있으면 OCR 시도 (선택적)
            # 여기에 OCR 로직 추가 가능 (예: Tesseract)
            # pix = page.get_pixmap()
            # img = Image.open(io.BytesIO(pix.tobytes("png")))
            # text = pytesseract.image_to_string(img, lang='kor+eng')  # 예시
            pass  # OCR은 현재 구현에서 제외
        return text.strip()
    except Exception as e:
        print(f"      ⚠️ Error extracting text from page: {e}")
        return ""

def analyze_pdf_file_text_only(pdf_path):
    """PDF 파일에서 텍스트만 추출합니다."""
    print(f"📄 Extracting text from PDF: {pdf_path.name}")
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        results = []
        
        print(f"   📄 Found {total_pages} pages")
        
        for page_num in range(total_pages): # 모든 페이지 처리
            print(f"   📄 Processing page {page_num + 1}...")
            
            try:
                page = doc.load_page(page_num)
                extracted_text = extract_text_from_pdf_page(page)
                
                if extracted_text:
                    results.append({
                        "page": page_num + 1,
                        "text_content": extracted_text
                    })
                    print(f"      ✅ Text extracted from page {page_num + 1}")
                else:
                    results.append({
                        "page": page_num + 1,
                        "text_content": "",
                        "warning": "No text found or extraction failed"
                    })
                    print(f"      ⚠️ No text extracted from page {page_num + 1}")
            except Exception as page_error:
                results.append({
                    "page": page_num + 1,
                    "text_content": "",
                    "error": str(page_error)
                })
                print(f"      ❌ Error processing page {page_num + 1}: {page_error}")
        
        doc.close()
        
        return {
            "file": pdf_path.name,
            "total_pages": total_pages,
            "processed_pages": len(results),
            "pages_text": results # 'pages' -> 'pages_text'로 변경
        }
        
    except Exception as e:
        print(f"   ❌ Error processing PDF for text extraction: {e}")
        return {"file": pdf_path.name, "error": str(e)}

def main():
    print("🏗️  Uploads Folder Text Extraction (PDF Only)")
    print("=" * 40)
    
    # uploads 폴더에서 프로젝트별 파일 찾기
    projects_to_analyze = find_files_in_uploads()
    
    if not projects_to_analyze:
        print("❌ No projects or PDF files found in uploads/ folder or its subdirectories.")
        print("\\n📝 Instructions:")
        print("   1. Place your PDF architectural drawings in the uploads/ folder.")
        print("      You can organize files into subfolders (each subfolder is a project).")
        print("   2. Only PDF files will be processed for text extraction.")
        print("   3. Run this script again")
        return
    
    print("📁 Found projects and PDF files:")
    for project_name, files_info in projects_to_analyze.items():
        print(f"  Project: {project_name} (Path: {files_info['path']})")
        print(f"    📄 PDF files: {len(files_info['pdf_files'])}")
        # 이미지 파일 개수는 더 이상 출력하지 않음

    # 분석 결과 저장
    all_projects_results = {}

    # 프로젝트별 분석 수행
    for project_name, files_info in projects_to_analyze.items():
        print(f"\\n\\n--- Extracting Text for Project: {project_name} ---")
        project_results = {
            "project_path": files_info["path"],
            "pdf_files_text": [], # 'pdf_files' -> 'pdf_files_text'
            # "image_files"는 더 이상 사용하지 않음
            "summary": {}
        }

        # PDF 파일들 텍스트 추출
        for pdf_file_path in files_info["pdf_files"]:
            result = analyze_pdf_file_text_only(pdf_file_path)
            project_results["pdf_files_text"].append(result)
        
        # 이미지 파일 분석은 제거됨
        
        # 프로젝트 요약 정보
        project_summary = {
            "total_pdf_files": len(files_info["pdf_files"]),
            # "total_image_files": 0, # 이미지 파일은 0으로 고정 또는 제거
            "total_files_processed": len(files_info["pdf_files"]), # PDF만 처리
            "successful_text_extractions": 0,
            "failed_text_extractions": 0,
        }

        # 성공/실패 카운트
        for pdf_result in project_results["pdf_files_text"]:
            if "error" not in pdf_result:
                # 각 페이지의 텍스트 추출 성공 여부도 확인할 수 있으나, 여기서는 파일 단위로 집계
                # 모든 페이지에서 텍스트가 성공적으로 추출되었는지, 일부만 되었는지 등
                # 여기서는 파일 처리 자체의 성공/실패만 카운트
                project_summary["successful_text_extractions"] += 1
            else:
                 project_summary["failed_text_extractions"] += 1
        
        project_results["summary"] = project_summary
        all_projects_results[project_name] = project_results

    # 전체 결과 저장
    output_file = "uploads_analysis_results.json" # 파일 이름은 유지
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_projects_results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 All project text extraction results saved to: {output_file}")
    
    # 전체 요약 출력
    total_all_files_processed = 0
    total_successful_extractions = 0
    total_failed_extractions = 0
    for project_name, project_data in all_projects_results.items():
        print(f"\\n📊 SUMMARY for Project: {project_name}")
        summary = project_data.get("summary", {})
        print(f"   📄 Total PDF files processed: {summary.get('total_pdf_files',0)}")
        print(f"   ✅ Successful text extractions (files): {summary.get('successful_text_extractions',0)}")
        print(f"   ❌ Failed text extractions (files): {summary.get('failed_text_extractions',0)}")
        total_all_files_processed += summary.get('total_pdf_files',0)
        total_successful_extractions += summary.get('successful_text_extractions',0)
        total_failed_extractions += summary.get('failed_text_extractions',0)
    
    print("\\n" + "="*40)
    print("📊 OVERALL SUMMARY")
    print(f"   🏗️ Total Projects Processed: {len(all_projects_results)}")
    print(f"   📄 Total PDF Files Processed (all projects): {total_all_files_processed}")
    print(f"   ✅ Total Successful Text Extractions (files, all projects): {total_successful_extractions}")
    print(f"   ❌ Total Failed Text Extractions (files, all projects): {total_failed_extractions}")
    
    print("\\n🎉 Text extraction completed!")

if __name__ == "__main__":
    main()
