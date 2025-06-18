#!/usr/bin/env python3
"""
개별 메타데이터 파일들을 통합하여 프로젝트 메타데이터 파일을 생성하는 스크립트
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_individual_metadata(project_dir: str) -> List[Dict[str, Any]]:
    """
    프로젝트 디렉토리에서 개별 메타데이터 파일들을 로드합니다.
    
    Args:
        project_dir: 프로젝트 디렉토리 경로
        
    Returns:
        개별 메타데이터 리스트
    """
    metadata_files = []
    project_path = Path(project_dir)
    
    # *_metadata.json 패턴의 파일들을 찾습니다
    for metadata_file in project_path.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 파일 경로 정보 추가
            data['file_path'] = str(metadata_file)
            data['file_name'] = metadata_file.name
            
            metadata_files.append(data)
            print(f"로드됨: {metadata_file.name}")
            
        except Exception as e:
            print(f"오류 - {metadata_file.name}: {e}")
            continue
    
    return metadata_files

def create_unified_metadata(metadata_list: List[Dict[str, Any]], project_name: str) -> Dict[str, Any]:
    """
    개별 메타데이터들을 통합된 프로젝트 메타데이터로 변환합니다.
    build_rag_db.py가 기대하는 형식에 맞춰 생성합니다.
    
    Args:
        metadata_list: 개별 메타데이터 리스트
        project_name: 프로젝트 이름
        
    Returns:
        통합된 프로젝트 메타데이터
    """
    unified_metadata = {
        "project_name": project_name,
        "created_at": datetime.now().isoformat(),
        "total_documents": len(metadata_list),
        "drawings": []  # build_rag_db.py가 기대하는 키
    }
    
    # 프로젝트 레벨 정보 추출 (첫 번째 문서에서)
    if metadata_list:
        first_doc = metadata_list[0]
        if 'project' in first_doc:
            unified_metadata["project_info"] = first_doc['project']
        elif 'projectInfo' in first_doc:
            unified_metadata["project_info"] = first_doc['projectInfo']
        elif 'project_info' in first_doc:
            unified_metadata["project_info"] = first_doc['project_info']
    
    # 각 문서별 메타데이터를 drawings 형식으로 변환
    for i, metadata in enumerate(metadata_list):
        # 원본 파일명에서 .pdf 확장자를 가진 실제 파일명 찾기
        actual_file_name = metadata.get('fileName', '')
        if not actual_file_name:
            actual_file_name = metadata.get('drawingTitle', '')
        if not actual_file_name:
            # _metadata.json에서 추정
            base_name = metadata.get('file_name', '').replace('_metadata.json', '')
            actual_file_name = f"{base_name}.pdf"
        
        drawing_metadata = {
            "drawing_number": f"DWG-{i+1:03d}",  # 도면 번호 생성
            "file_name": actual_file_name,
            "page_number": 1,  # PDF는 일반적으로 1페이지
            "drawing_title": metadata.get('drawingTitle', metadata.get('drawing_title', actual_file_name)),
            "drawing_type": metadata.get('documentType', metadata.get('document_type', 'unknown')),
            "scale": "정보 없음",  # 축척 정보가 없으면 기본값
            "extracted_at": metadata.get('extractedAt', metadata.get('extracted_at', '')),
        }
        
        # 면적 정보 추가
        area_info = {}
        if 'project' in metadata and 'landAreaM2' in metadata['project']:
            area_info['대지면적'] = f"{metadata['project']['landAreaM2']}㎡"
        if 'project' in metadata and 'buildingAreaM2' in metadata['project']:
            area_info['건축면적'] = f"{metadata['project']['buildingAreaM2']}㎡"
        if 'project' in metadata and 'totalFloorAreaM2' in metadata['project']:
            area_info['연면적'] = f"{metadata['project']['totalFloorAreaM2']}㎡"
        
        if area_info:
            drawing_metadata['area_info'] = area_info
        
        # 공간 목록 추가 (시설 정보에서 추출)
        room_list = []
        if 'facilityArea' in metadata:
            room_list = list(metadata['facilityArea'].keys())
        elif 'facility_area' in metadata:
            room_list = list(metadata['facility_area'].keys())
        
        if room_list:
            drawing_metadata['room_list'] = room_list
        
        # 층 정보 추가
        level_info = []
        if 'project' in metadata and 'floors' in metadata['project']:
            floors = metadata['project']['floors']
            if 'above' in floors:
                level_info.append(f"지상 {floors['above']}층")
            if 'below' in floors:
                level_info.append(f"지하 {floors['below']}층")
        
        if level_info:
            drawing_metadata['level_info'] = level_info
        
        # 텍스트 스니펫 생성 (검색 향상을 위해)
        text_parts = []
        if 'project' in metadata:
            proj = metadata['project']
            if 'projectName' in proj:
                text_parts.append(f"프로젝트명: {proj['projectName']}")
            if 'siteLocation' in proj:
                text_parts.append(f"위치: {proj['siteLocation']}")
            if 'structureType' in proj:
                text_parts.append(f"구조: {proj['structureType']}")
            if 'unitType' in proj:
                text_parts.append(f"용도: {proj['unitType']}")
        
        if 'unitCount' in metadata:
            text_parts.append(f"세대수: {metadata['unitCount']}세대")
        
        if 'parking' in metadata:
            parking = metadata['parking']
            total_parking = sum([v for v in parking.values() if isinstance(v, int)])
            text_parts.append(f"주차대수: {total_parking}대")
        
        if text_parts:
            drawing_metadata['raw_text_snippet'] = '. '.join(text_parts)
        
        # 원본 메타데이터 보존
        drawing_metadata['original_metadata'] = metadata
        
        unified_metadata["drawings"].append(drawing_metadata)
    
    return unified_metadata

def save_unified_metadata(unified_metadata: Dict[str, Any], output_path: str):
    """
    통합된 메타데이터를 파일로 저장합니다.
    
    Args:
        unified_metadata: 통합된 메타데이터
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"통합 메타데이터 저장됨: {output_path}")

def main():
    """메인 함수"""
    project_name = "부산장안지구"
    project_dir = f"uploads/{project_name}"
    output_file = f"uploads/{project_name}/project_metadata_{project_name}.json"
    
    print(f"프로젝트: {project_name}")
    print(f"디렉토리: {project_dir}")
    print("-" * 50)
    
    # 개별 메타데이터 파일들 로드
    metadata_list = load_individual_metadata(project_dir)
    
    if not metadata_list:
        print("메타데이터 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n총 {len(metadata_list)}개의 메타데이터 파일을 찾았습니다.")
    
    # 통합 메타데이터 생성
    unified_metadata = create_unified_metadata(metadata_list, project_name)
    
    # 파일로 저장
    save_unified_metadata(unified_metadata, output_file)
    
    # 결과 요약 출력
    print("\n=== 통합 결과 요약 ===")
    print(f"프로젝트명: {unified_metadata['project_name']}")
    print(f"문서 수: {unified_metadata['total_documents']}")
    print(f"생성일시: {unified_metadata['created_at']}")
    
    print("\n포함된 도면들:")
    for drawing in unified_metadata['drawings']:
        print(f"  - {drawing['drawing_title']} ({drawing['drawing_type']}) - {drawing['drawing_number']}")

if __name__ == "__main__":
    main()
