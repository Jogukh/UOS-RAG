#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
건축 PDF RAG 시스템 통합 워크플로우

전체 파이프라인:
1. DWG/PDF 파일에서 데이터 추출
2. LLM 기반 전처리 및 JSON (Self-Query 형태) 파싱
3. RAG DB 생성 (ChromaDB)
4. 챗봇 시스템에서 질의/응답

사용법:
    python workflow_pipeline.py --mode=extract --project_name="부산장안지구"
    python workflow_pipeline.py --mode=build_db --project_name="부산장안지구"
    python workflow_pipeline.py --mode=query --query="건축면적이 얼마인가요?" --project="부산장안지구"
    python workflow_pipeline.py --mode=full --project_name="부산장안지구"
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

def run_command(command: str, description: str) -> bool:
    """명령어 실행 및 결과 반환"""
    print(f"\n🔄 {description}")
    print(f"   실행: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패:")
        print(f"   오류: {e.stderr}")
        return False

def extract_metadata(project_name: str, file_types: str = "pdf,dwg") -> bool:
    """1단계: DWG/PDF 파일에서 메타데이터 추출"""
    command = f"python extract_metadata_unified.py --project_name='{project_name}' --file_types='{file_types}'"
    return run_command(command, f"메타데이터 추출 (프로젝트: {project_name}, 형식: {file_types})")

def build_rag_db(project_name: Optional[str] = None) -> bool:
    """2단계: RAG 데이터베이스 구축 (Self-Query 변환 포함)"""
    if project_name:
        command = f"python build_rag_db.py --project_name='{project_name}'"
        description = f"RAG DB 구축 (프로젝트: {project_name})"
    else:
        command = "python build_rag_db.py"
        description = "RAG DB 구축 (전체 프로젝트)"
    
    return run_command(command, description)

def query_rag(query: str, project: Optional[str] = None) -> bool:
    """3단계: RAG 시스템 질의"""
    if project:
        command = f"python query_rag.py '{query}' --project='{project}'"
        description = f"RAG 질의 (프로젝트: {project})"
    else:
        command = f"python query_rag.py '{query}'"
        description = "RAG 질의 (전체 프로젝트)"
    
    return run_command(command, description)

def full_pipeline(project_name: str, file_types: str = "pdf,dwg") -> bool:
    """전체 파이프라인 실행"""
    print(f"\n🚀 전체 파이프라인 시작 (프로젝트: {project_name}, 형식: {file_types})")
    
    # 1단계: 메타데이터 추출
    if not extract_metadata(project_name, file_types):
        print("❌ 메타데이터 추출 실패. 파이프라인을 중단합니다.")
        return False
    
    # 2단계: RAG DB 구축
    if not build_rag_db(project_name):
        print("❌ RAG DB 구축 실패. 파이프라인을 중단합니다.")
        return False
    
    print(f"\n✅ 전체 파이프라인 완료 (프로젝트: {project_name})")
    print(f"💡 이제 다음 명령으로 질의할 수 있습니다:")
    print(f"   python query_rag.py \"건축면적이 얼마인가요?\" --project=\"{project_name}\"")
    
    return True

def list_projects() -> bool:
    """사용 가능한 프로젝트 목록 표시"""
    command = "python query_rag.py --list_projects"
    return run_command(command, "프로젝트 목록 조회")

def main():
    parser = argparse.ArgumentParser(
        description="건축 PDF RAG 시스템 통합 워크플로우",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행
  python workflow_pipeline.py --mode=full --project_name="부산장안지구"
  
  # 단계별 실행
  python workflow_pipeline.py --mode=extract --project_name="부산장안지구"
  python workflow_pipeline.py --mode=build_db --project_name="부산장안지구"
  python workflow_pipeline.py --mode=query --query="건축면적이 얼마인가요?" --project="부산장안지구"
  
  # 프로젝트 목록 확인
  python workflow_pipeline.py --mode=list_projects
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["extract", "build_db", "query", "full", "list_projects"],
        required=True,
        help="실행할 모드 선택"
    )
    
    parser.add_argument(
        "--project_name",
        type=str,
        help="프로젝트 이름 (extract, build_db, full 모드에서 사용)"
    )
    
    parser.add_argument(
        "--file_types",
        type=str,
        default="pdf,dwg",
        help="처리할 파일 형식 (pdf, dwg, 또는 pdf,dwg). 기본값: pdf,dwg"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        help="질의 대상 프로젝트 (query 모드에서 사용)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="질의 문장 (query 모드에서 사용)"
    )
    
    args = parser.parse_args()
    
    # 모드별 실행
    if args.mode == "extract":
        if not args.project_name:
            print("❌ extract 모드에서는 --project_name이 필요합니다.")
            sys.exit(1)
        success = extract_metadata(args.project_name, args.file_types)
        
    elif args.mode == "build_db":
        success = build_rag_db(args.project_name)
        
    elif args.mode == "query":
        if not args.query:
            print("❌ query 모드에서는 --query가 필요합니다.")
            sys.exit(1)
        success = query_rag(args.query, args.project)
        
    elif args.mode == "full":
        if not args.project_name:
            print("❌ full 모드에서는 --project_name이 필요합니다.")
            sys.exit(1)
        success = full_pipeline(args.project_name, args.file_types)
        
    elif args.mode == "list_projects":
        success = list_projects()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
