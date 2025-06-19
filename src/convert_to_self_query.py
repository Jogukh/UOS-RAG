#!/usr/bin/env python3
"""
기존 메타데이터를 Self-Query 형식으로 변환하는 유틸리티
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

def extract_numeric_value(text: str) -> Optional[float]:
    """텍스트에서 숫자 값 추출"""
    if not text or text == "정보 없음":
        return None
    
    # 숫자와 소수점만 추출
    numbers = re.findall(r'\d+\.?\d*', str(text))
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None

def extract_floor_level(text: str) -> Optional[int]:
    """층수 정보를 정수로 변환"""
    if not text:
        return None
    
    text = str(text).lower()
    if '지하' in text:
        numbers = re.findall(r'\d+', text)
        if numbers:
            return -int(numbers[0])  # 지하는 음수
    elif '층' in text or 'floor' in text:
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
    
    return None

def extract_year(text: str) -> Optional[int]:
    """연도 정보 추출"""
    if not text:
        return None
    
    # YYYY 형식의 연도 찾기
    years = re.findall(r'20\d{2}', str(text))
    if years:
        return int(years[0])
    
    return None

def count_elements(data: Any) -> int:
    """리스트나 문자열에서 요소 개수 계산"""
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, str) and data:
        # 콤마로 구분된 항목들 계산
        return len([x.strip() for x in data.split(',') if x.strip()])
    return 0

def calculate_completion_score(metadata: Dict[str, Any]) -> int:
    """메타데이터 완성도 점수 계산 (0-100)"""
    total_fields = 20  # 주요 필드 개수
    filled_fields = 0
    
    important_fields = [
        'drawing_number', 'drawing_title', 'drawing_type', 'project_name',
        'exclusive_area', 'supply_area', 'floor_level', 'unit_type'
    ]
    
    for field in important_fields:
        if field in metadata and metadata[field] not in [None, "", "정보 없음", "unknown"]:
            filled_fields += 1
    
    # 추가 정보가 있으면 보너스 점수
    bonus = 0
    if metadata.get('has_tables'):
        bonus += 10
    if metadata.get('has_dimensions'):
        bonus += 10
    if metadata.get('material_count', 0) > 0:
        bonus += 5
    if metadata.get('room_count', 0) > 0:
        bonus += 5
    
    base_score = int((filled_fields / len(important_fields)) * 70)
    return min(100, base_score + bonus)

def convert_to_self_query_format(old_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """기존 메타데이터를 Self-Query 형식으로 변환"""
    
    # 기본 정보 추출
    basic_info = old_metadata.get('basic_info', {}) or {}
    project_info = basic_info.get('project_info', {}) or {}
    spatial_info = old_metadata.get('spatial_info', {}) or {}
    technical_info = old_metadata.get('technical_info', {}) or {}
    total_area = spatial_info.get('total_area', {}) or {}
    quality_indicators = old_metadata.get('quality_indicators', {}) or {}
    
    # Self-Query 메타데이터 구성
    metadata = {
        "drawing_number": basic_info.get('drawing_number', ''),
        "drawing_title": basic_info.get('drawing_title', ''),
        "drawing_type": basic_info.get('drawing_type', ''),
        "scale": basic_info.get('scale', ''),
        "project_name": project_info.get('project_name', ''),
        "location": project_info.get('location', ''),
        "building_type": project_info.get('building_type', ''),
        "design_office": project_info.get('design_office', ''),
        "architect": project_info.get('architect', ''),
    }
    
    # 숫자 필드 변환
    metadata["floor_level"] = extract_floor_level(spatial_info.get('floor_level', ''))
    metadata["design_year"] = extract_year(project_info.get('date', ''))
    metadata["exclusive_area"] = extract_numeric_value(total_area.get('exclusive_area', ''))
    metadata["supply_area"] = extract_numeric_value(total_area.get('supply_area', ''))
    metadata["site_area"] = extract_numeric_value(total_area.get('site_area', ''))
    
    # 세대형 정보 정리
    unit_type = spatial_info.get('unit_type', '')
    if unit_type and '㎡' in unit_type:
        # "59㎡A형" -> "59A"
        unit_type = re.sub(r'㎡.*형?', '', unit_type)
    metadata["unit_type"] = unit_type
    
    # 불린 필드
    metadata["has_tables"] = bool(old_metadata.get('tables_extracted')) or \
                            bool(quality_indicators.get('has_tables'))
    metadata["has_dimensions"] = bool(technical_info.get('dimensions'))
    
    # 계산 필드
    metadata["material_count"] = count_elements(technical_info.get('materials', []))
    metadata["room_count"] = count_elements(spatial_info.get('rooms', []))
    metadata["completion_score"] = calculate_completion_score(metadata)
    
    # content 생성 (검색 텍스트)
    content_parts = []
    
    if metadata["project_name"]:
        content_parts.append(f"프로젝트: {metadata['project_name']}")
    if metadata["drawing_type"]:
        content_parts.append(f"도면유형: {metadata['drawing_type']}")
    if metadata["drawing_title"]:
        content_parts.append(f"도면명: {metadata['drawing_title']}")
    if metadata["exclusive_area"]:
        content_parts.append(f"전용면적: {metadata['exclusive_area']}㎡")
    if metadata["floor_level"]:
        level_str = f"{metadata['floor_level']}층" if metadata['floor_level'] > 0 else f"지하{abs(metadata['floor_level'])}층"
        content_parts.append(f"층수: {level_str}")
    
    # 실 정보 추가
    rooms = spatial_info.get('rooms', [])
    if rooms:
        room_names = [room.get('name', '') for room in rooms if room.get('name')]
        if room_names:
            content_parts.append(f"주요공간: {', '.join(room_names)}")
    
    # 재료 정보 추가
    materials = technical_info.get('materials', [])
    if materials:
        content_parts.append(f"재료: {', '.join(materials[:3])}")  # 처음 3개만
    
    content = ". ".join(content_parts) + "."
    
    # 추가 배열 데이터
    result = {
        "content": content,
        "metadata": {k: v for k, v in metadata.items() if v not in [None, "", "정보 없음", "unknown"]},
        "rooms": spatial_info.get('rooms', []),
        "materials": technical_info.get('materials', []),
        "dimensions": technical_info.get('dimensions', []),
        "equipment_codes": technical_info.get('equipment_codes', []),
        "notes": old_metadata.get('notes_and_symbols', [])
    }
    
    return result

def convert_project_metadata_file(input_file: str, output_file: str = None):
    """프로젝트 메타데이터 파일을 Self-Query 형식으로 변환"""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
    
    if output_file is None:
        output_file = input_path.parent / f"selfquery_{input_path.name}"
    
    # 기존 메타데이터 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        project_data = json.load(f)
    
    converted_drawings = []
    
    if 'drawings' in project_data:
        for drawing in project_data['drawings']:
            converted = convert_to_self_query_format(drawing)
            converted_drawings.append(converted)
    
    # Self-Query 형식으로 저장
    output_data = {
        "project_name": project_data.get('project_name', ''),
        "total_drawings": len(converted_drawings),
        "conversion_date": "2025-06-19",
        "format": "self_query_compatible",
        "drawings": converted_drawings
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Self-Query 형식 변환 완료:")
    print(f"   입력: {input_file}")
    print(f"   출력: {output_file}")
    print(f"   변환된 도면 수: {len(converted_drawings)}")
    
    return output_file

if __name__ == "__main__":
    # 테스트 실행
    input_file = "uploads/부산장안지구/project_metadata_부산장안지구.json"
    
    if Path(input_file).exists():
        convert_project_metadata_file(input_file)
    else:
        print(f"테스트 파일이 없습니다: {input_file}")
        print("사용법: python convert_to_self_query.py")
