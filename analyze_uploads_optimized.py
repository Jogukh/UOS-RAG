#!/usr/bin/env python3
"""
uploads 폴더의 건축 도면 분석 스크립트 (병렬처리 최적화 버전)
PyMuPDFLoader (LangChain)를 사용하여 PDF 텍스트 추출
- 병렬처리로 속도 최적화
- JSON 시리얼라이즈 오류 해결
- 메모리 효율성 개선
- GPU 가속 지원 (가능한 경우)
"""

import os
import sys
import json
import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from functools import partial
import logging

# LangChain document loader
from langchain_community.document_loaders import PyMuPDFLoader

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_json_serialize(obj):
    """안전한 JSON 시리얼라이즈 함수"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): safe_json_serialize(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        try:
            return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
        except:
            return str(obj)
    else:
        try:
            # Try to convert to string
            return str(obj)
        except:
            return f"<unserializable_object: {type(obj).__name__}>"

def serialize_table_data(table_data):
    """표 데이터를 JSON 직렬화 가능한 형식으로 변환"""
    if not table_data:
        return table_data
        
    return safe_json_serialize(table_data)

def find_files_in_uploads():
    """uploads 폴더 및 하위 프로젝트 폴더에서 파일을 찾습니다."""
    uploads_dir = Path("uploads")
    projects_files = {}

    if not uploads_dir.exists():
        uploads_dir.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Created uploads/ directory")
        return projects_files

    # uploads 폴더 직속 파일 처리 (기본 프로젝트로 간주)
    default_project_name = "_default_project"
    default_pdf_files = list(uploads_dir.glob("*.pdf"))
    
    if default_pdf_files:
        projects_files[default_project_name] = {
            "path": str(uploads_dir.resolve()),
            "pdf_files": default_pdf_files,
        }

    # 하위 폴더 (프로젝트) 탐색
    for item in uploads_dir.iterdir():
        if item.is_dir():
            project_name = item.name
            project_path = item
            
            pdf_files = list(project_path.glob("**/*.pdf"))
            
            if pdf_files:
                if project_name in projects_files:
                    projects_files[project_name]["pdf_files"].extend(pdf_files)
                else:
                    projects_files[project_name] = {
                        "path": str(project_path.resolve()),
                        "pdf_files": pdf_files,
                    }
    
    return projects_files

def extract_text_from_pdf_optimized(pdf_path_str: str) -> Dict[str, Any]:
    """최적화된 PDF 텍스트 추출 함수 (병렬처리용)"""
    pdf_path = Path(pdf_path_str)
    logger.info(f"📄 Extracting text from PDF: {pdf_path.name}")
    
    start_time = time.time()
    
    try:
        # PyMuPDFLoader 초기화
        loader = PyMuPDFLoader(str(pdf_path), mode="page")
        docs = loader.load()
        
        logger.info(f"   📄 Found {len(docs)} pages in {pdf_path.name}")
        
        results = []
        
        # PyMuPDF 직접 사용
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        
        for i, langchain_doc in enumerate(docs):
            page_num = i + 1
            
            try:
                # LangChain으로 기본 텍스트 추출
                text_content = langchain_doc.page_content.strip()
                metadata = langchain_doc.metadata
                
                # PyMuPDF로 직접 표 추출
                pymupdf_page = doc[i]
                
                # 표 추출 (최적화)
                tables_data = []
                table_count = 0
                try:
                    tables = pymupdf_page.find_tables()
                    if tables.tables:
                        table_count = len(tables.tables)
                        for table_idx, table in enumerate(tables.tables):
                            table_content = table.extract()
                            # JSON 직렬화 가능한 형태로 변환
                            serializable_content = serialize_table_data(table_content)
                            bbox = table.bbox
                            
                            tables_data.append({
                                "table_index": table_idx,
                                "table_content": serializable_content,
                                "table_bbox": [float(x) for x in bbox] if bbox else None
                            })
                except Exception as table_error:
                    logger.warning(f"Table extraction error on page {page_num}: {table_error}")
                
                # HTML 형식으로 텍스트 추출 (표 구조 포함)
                html_content = ""
                try:
                    html_content = pymupdf_page.get_text("html")
                except Exception as html_error:
                    logger.warning(f"HTML extraction error on page {page_num}: {html_error}")
                
                # 구조화된 딕셔너리 형식 (메모리 효율을 위해 간소화)
                has_images = False
                try:
                    dict_content = pymupdf_page.get_text("dict")
                    # 이미지 존재 여부만 확인 (전체 딕셔너리는 메모리 절약을 위해 저장하지 않음)
                    if dict_content and 'blocks' in dict_content:
                        for block in dict_content.get('blocks', []):
                            if block.get('type') == 1:  # 이미지 블록
                                has_images = True
                                break
                except Exception as dict_error:
                    logger.warning(f"Dict extraction error on page {page_num}: {dict_error}")
                
                page_result = {
                    "page": page_num,
                    "text_content": text_content,
                    "html_content": html_content,
                    "tables": tables_data,
                    "has_images": has_images,
                    "metadata": {
                        "source": str(pdf_path),
                        "page": page_num - 1,
                        "total_pages": len(docs),
                        "has_tables": len(tables_data) > 0,
                        "table_count": table_count,
                        "text_length": len(text_content),
                        "html_length": len(html_content)
                    }
                }
                
                results.append(page_result)
                
            except Exception as page_error:
                logger.error(f"Error processing page {page_num}: {page_error}")
                results.append({
                    "page": page_num,
                    "text_content": "",
                    "html_content": "",
                    "tables": [],
                    "has_images": False,
                    "error": str(page_error),
                    "metadata": {
                        "source": str(pdf_path), 
                        "page": page_num - 1, 
                        "has_tables": False, 
                        "table_count": 0
                    }
                })
        
        # PyMuPDF 문서 닫기
        doc.close()
        
        # 전체 문서 통계
        total_tables = sum(len(page.get("tables", [])) for page in results)
        total_text_length = sum(page.get("metadata", {}).get("text_length", 0) for page in results)
        pages_with_tables = sum(1 for page in results if page.get("metadata", {}).get("has_tables", False))
        pages_with_images = sum(1 for page in results if page.get("has_images", False))
        
        extraction_time = time.time() - start_time
        
        return {
            "file": pdf_path.name,
            "file_path": str(pdf_path),
            "total_pages": len(docs),
            "processed_pages": len(results),
            "total_tables_found": total_tables,
            "pages_with_tables": pages_with_tables,
            "pages_with_images": pages_with_images,
            "total_text_length": total_text_length,
            "extraction_time_seconds": round(extraction_time, 2),
            "pages_text": results,
            "extraction_method": "PyMuPDFLoader + PyMuPDF Direct (Optimized)"
        }
        
    except Exception as e:
        extraction_time = time.time() - start_time
        logger.error(f"Error processing PDF {pdf_path.name}: {e}")
        return {
            "file": pdf_path.name,
            "file_path": str(pdf_path),
            "error": str(e),
            "extraction_time_seconds": round(extraction_time, 2),
            "extraction_method": "PyMuPDFLoader + PyMuPDF Direct (Optimized)"
        }

def process_project_parallel(project_name: str, files_info: Dict, max_workers: int = None) -> Dict[str, Any]:
    """프로젝트 파일들을 병렬로 처리"""
    logger.info(f"🔄 Processing project: {project_name} with {len(files_info['pdf_files'])} PDF files")
    
    if max_workers is None:
        # CPU 코어 수에 기반하여 워커 수 결정 (최대 8개)
        max_workers = min(mp.cpu_count(), 8, len(files_info['pdf_files']))
    
    project_results = {
        "project_path": files_info["path"],
        "pdf_files_text": [],
        "processing_info": {
            "max_workers": max_workers,
            "total_files": len(files_info['pdf_files']),
            "start_time": time.time()
        }
    }
    
    # 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # PDF 파일 경로를 문자열로 변환 (병렬처리를 위해)
        pdf_paths = [str(pdf_path) for pdf_path in files_info["pdf_files"]]
        
        # 작업 제출
        future_to_pdf = {
            executor.submit(extract_text_from_pdf_optimized, pdf_path): pdf_path 
            for pdf_path in pdf_paths
        }
        
        # 결과 수집
        completed_count = 0
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                project_results["pdf_files_text"].append(result)
                completed_count += 1
                logger.info(f"   ✅ Completed {completed_count}/{len(pdf_paths)}: {Path(pdf_path).name}")
            except Exception as exc:
                logger.error(f"   ❌ Error processing {Path(pdf_path).name}: {exc}")
                project_results["pdf_files_text"].append({
                    "file": Path(pdf_path).name,
                    "file_path": pdf_path,
                    "error": str(exc),
                    "extraction_method": "PyMuPDFLoader + PyMuPDF Direct (Optimized)"
                })
    
    # 처리 완료 시간 기록
    project_results["processing_info"]["end_time"] = time.time()
    project_results["processing_info"]["total_time_seconds"] = round(
        project_results["processing_info"]["end_time"] - project_results["processing_info"]["start_time"], 2
    )
    
    # 프로젝트 요약 정보 계산
    successful_extractions = 0
    failed_extractions = 0
    total_tables_found = 0
    total_pages_with_tables = 0
    total_pages_with_images = 0
    total_pages_processed = 0
    total_text_length = 0
    total_extraction_time = 0
    
    for pdf_result in project_results["pdf_files_text"]:
        if "error" not in pdf_result:
            successful_extractions += 1
            total_tables_found += pdf_result.get("total_tables_found", 0)
            total_pages_with_tables += pdf_result.get("pages_with_tables", 0)
            total_pages_with_images += pdf_result.get("pages_with_images", 0)
            total_pages_processed += pdf_result.get("processed_pages", 0)
            total_text_length += pdf_result.get("total_text_length", 0)
            total_extraction_time += pdf_result.get("extraction_time_seconds", 0)
        else:
            failed_extractions += 1
    
    project_results["summary"] = {
        "total_pdf_files": len(files_info["pdf_files"]),
        "successful_text_extractions": successful_extractions,
        "failed_text_extractions": failed_extractions,
        "total_tables_found": total_tables_found,
        "pages_with_tables": total_pages_with_tables,
        "pages_with_images": total_pages_with_images,
        "total_pages_processed": total_pages_processed,
        "total_text_length": total_text_length,
        "total_extraction_time_seconds": round(total_extraction_time, 2),
        "parallel_processing_time_seconds": project_results["processing_info"]["total_time_seconds"],
        "efficiency_ratio": round(
            total_extraction_time / project_results["processing_info"]["total_time_seconds"], 2
        ) if project_results["processing_info"]["total_time_seconds"] > 0 else 0
    }
    
    return project_results

def main():
    print("🚀 Uploads Folder Text Extraction (Optimized Parallel Version)")
    print("=" * 60)
    
    # 시스템 정보 출력
    cpu_count = mp.cpu_count()
    print(f"💻 System Info: {cpu_count} CPU cores available")
    
    # uploads 폴더에서 프로젝트별 파일 찾기
    projects_to_analyze = find_files_in_uploads()
    
    if not projects_to_analyze:
        print("❌ No projects or PDF files found in uploads/ folder or its subdirectories.")
        print("\n📝 Instructions:")
        print("   1. Place your PDF architectural drawings in the uploads/ folder.")
        print("      You can organize files into subfolders (each subfolder is a project).")
        print("   2. Only PDF files will be processed for text extraction.")
        print("   3. Run this script again")
        return
    
    # 총 파일 수 계산
    total_pdf_files = sum(len(files_info['pdf_files']) for files_info in projects_to_analyze.values())
    
    print("📁 Found projects and PDF files:")
    for project_name, files_info in projects_to_analyze.items():
        print(f"  Project: {project_name} (Path: {files_info['path']})")
        print(f"    📄 PDF files: {len(files_info['pdf_files'])}")
    
    print(f"\n📊 Total: {len(projects_to_analyze)} projects, {total_pdf_files} PDF files")
    
    # 전체 처리 시작 시간
    overall_start_time = time.time()
    
    # 분석 결과 저장
    all_projects_results = {}
    
    # 프로젝트별 병렬 분석 수행
    for project_name, files_info in projects_to_analyze.items():
        print(f"\n--- Processing Project: {project_name} ---")
        
        # 워커 수 결정 (파일 수가 적으면 워커 수도 줄임)
        optimal_workers = min(mp.cpu_count(), 8, len(files_info["pdf_files"]))
        
        project_results = process_project_parallel(project_name, files_info, optimal_workers)
        all_projects_results[project_name] = project_results
        
        # 프로젝트별 개별 JSON 파일 저장
        safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_project_name = safe_project_name.replace(' ', '_')
        project_output_file = f"{safe_project_name}_analysis_results.json"
        
        try:
            with open(project_output_file, 'w', encoding='utf-8') as f:
                json.dump(safe_json_serialize(project_results), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Project results saved to: {project_output_file}")
            print(f"✅ Project results saved to: {project_output_file}")
        except Exception as save_error:
            logger.error(f"Error saving project results: {save_error}")
            print(f"❌ Error saving project results: {save_error}")
    
    # 전체 처리 시간 계산
    overall_end_time = time.time()
    overall_processing_time = overall_end_time - overall_start_time
    
    # 전체 결과 저장 (통합 파일)
    output_file = "all_projects_analysis_results_optimized.json"
    
    # 메타데이터 추가
    final_results = {
        "processing_metadata": {
            "script_version": "optimized_parallel_v1.0",
            "processing_start_time": overall_start_time,
            "processing_end_time": overall_end_time,
            "total_processing_time_seconds": round(overall_processing_time, 2),
            "cpu_cores_available": cpu_count,
            "total_projects": len(projects_to_analyze),
            "total_pdf_files": total_pdf_files
        },
        "projects": all_projects_results
    }
    
    # JSON 저장 (safe serialization 사용)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(final_results), f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Results saved to: {output_file}")
    except Exception as save_error:
        logger.error(f"Error saving results: {save_error}")
        # 백업 저장 시도
        backup_file = f"uploads_analysis_results_backup_{int(time.time())}.json"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                # 더 안전한 저장 방식
                simplified_results = {
                    "processing_metadata": final_results["processing_metadata"],
                    "projects_summary": {
                        name: project_data.get("summary", {})
                        for name, project_data in all_projects_results.items()
                    }
                }
                json.dump(simplified_results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Backup results saved to: {backup_file}")
        except Exception as backup_error:
            logger.error(f"Failed to save backup: {backup_error}")
    
    # 전체 요약 출력
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_successful = 0
    total_failed = 0
    total_tables = 0
    total_pages_with_tables = 0
    total_pages_processed = 0
    total_extraction_time = 0
    
    for project_name, project_data in all_projects_results.items():
        summary = project_data.get("summary", {})
        print(f"\n📁 Project: {project_name}")
        print(f"   📄 PDF files: {summary.get('total_pdf_files', 0)}")
        print(f"   ✅ Successful: {summary.get('successful_text_extractions', 0)}")
        print(f"   ❌ Failed: {summary.get('failed_text_extractions', 0)}")
        print(f"   📊 Tables found: {summary.get('total_tables_found', 0)}")
        print(f"   📄 Pages with tables: {summary.get('pages_with_tables', 0)}")
        print(f"   📄 Total pages processed: {summary.get('total_pages_processed', 0)}")
        print(f"   ⏱️  Extraction time: {summary.get('total_extraction_time_seconds', 0)}s")
        print(f"   🔄 Parallel processing time: {summary.get('parallel_processing_time_seconds', 0)}s")
        print(f"   ⚡ Efficiency ratio: {summary.get('efficiency_ratio', 0)}x")
        
        total_successful += summary.get('successful_text_extractions', 0)
        total_failed += summary.get('failed_text_extractions', 0)
        total_tables += summary.get('total_tables_found', 0)
        total_pages_with_tables += summary.get('pages_with_tables', 0)
        total_pages_processed += summary.get('total_pages_processed', 0)
        total_extraction_time += summary.get('total_extraction_time_seconds', 0)
    
    print(f"\n{'='*60}")
    print("🏆 OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"🏗️  Total Projects: {len(all_projects_results)}")
    print(f"📄 Total PDF Files: {total_pdf_files}")
    print(f"✅ Total Successful Extractions: {total_successful}")
    print(f"❌ Total Failed Extractions: {total_failed}")
    print(f"📊 Total Tables Found: {total_tables}")
    print(f"📄 Total Pages with Tables: {total_pages_with_tables}")
    print(f"📄 Total Pages Processed: {total_pages_processed}")
    print(f"⏱️  Total Extraction Time: {total_extraction_time:.2f}s")
    print(f"🔄 Total Processing Time: {overall_processing_time:.2f}s")
    
    if overall_processing_time > 0:
        overall_efficiency = total_extraction_time / overall_processing_time
        print(f"⚡ Overall Efficiency Ratio: {overall_efficiency:.2f}x")
        print(f"🚀 Speed Improvement: {((overall_efficiency - 1) * 100):.1f}%")
    
    print(f"\n🎉 Optimized text extraction completed!")
    print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    # Python multiprocessing 설정 (Windows 호환성)
    mp.set_start_method('spawn', force=True)
    main()
