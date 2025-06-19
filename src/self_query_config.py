#!/usr/bin/env python3
"""
Self-Query Retriever를 위한 건축 도면 메타데이터 설정
"""

from langchain.chains.query_constructor.schema import AttributeInfo

# 건축 도면 메타데이터 필드 정의
ARCHITECTURAL_METADATA_FIELDS = [
    AttributeInfo(
        name="drawing_number",
        description="도면번호 (예: A04-001, S01-002)",
        type="string"
    ),
    AttributeInfo(
        name="drawing_title", 
        description="도면제목",
        type="string"
    ),
    AttributeInfo(
        name="drawing_type",
        description="도면유형. 다음 중 하나: 평면도, 입면도, 단면도, 상세도, 배치도, 일람표, 개요",
        type="string"
    ),
    AttributeInfo(
        name="scale",
        description="축척 정보 (예: 1:40, 1:80, 1:100)",
        type="string"
    ),
    AttributeInfo(
        name="project_name",
        description="프로젝트명",
        type="string"
    ),
    AttributeInfo(
        name="location",
        description="건물이 위치한 지역",
        type="string"
    ),
    AttributeInfo(
        name="building_type",
        description="건물 유형 (예: 공동주택, 업무시설, 근린생활시설)",
        type="string"
    ),
    AttributeInfo(
        name="floor_level",
        description="층수. 지상층은 양수, 지하층은 음수로 표현 (예: 1, 2, -1, -2)",
        type="integer"
    ),
    AttributeInfo(
        name="unit_type", 
        description="세대형 (예: 59A, 84B, 110C)",
        type="string"
    ),
    AttributeInfo(
        name="exclusive_area",
        description="전용면적 (제곱미터 단위의 숫자)",
        type="float"
    ),
    AttributeInfo(
        name="supply_area",
        description="공급면적 (제곱미터 단위의 숫자)",
        type="float"
    ),
    AttributeInfo(
        name="site_area",
        description="대지면적 (제곱미터 단위의 숫자)",
        type="float"
    ),
    AttributeInfo(
        name="design_year",
        description="설계 연도 (예: 2023, 2024)",
        type="integer"
    ),
    AttributeInfo(
        name="design_office",
        description="설계사무소명",
        type="string"
    ),
    AttributeInfo(
        name="architect",
        description="건축사명",
        type="string"
    ),
    AttributeInfo(
        name="has_tables",
        description="도면에 표(테이블) 포함 여부",
        type="boolean"
    ),
    AttributeInfo(
        name="has_dimensions",
        description="도면에 치수 정보 포함 여부",
        type="boolean"
    ),
    AttributeInfo(
        name="material_count",
        description="도면에 포함된 재료 종류의 개수",
        type="integer"
    ),
    AttributeInfo(
        name="room_count",
        description="도면에 표시된 실(방)의 개수",
        type="integer"
    ),
    AttributeInfo(
        name="completion_score",
        description="메타데이터 정보 완성도 점수 (0-100)",
        type="integer"
    )
]

# 문서 내용 설명
DOCUMENT_CONTENT_DESCRIPTION = """건축 도면의 주요 내용과 특징을 설명하는 텍스트. 
도면 유형, 주요 공간, 면적 정보, 재료, 치수 등이 자연어로 요약되어 있음"""

# Self-Query에서 사용할 컬렉션명 패턴
COLLECTION_NAME_PATTERN = "drawings_{project_name}"

# 지원하는 검색 연산자
SUPPORTED_OPERATORS = {
    "comparison": ["eq", "ne", "gt", "gte", "lt", "lte", "contain", "like", "in", "nin"],
    "logical": ["and", "or", "not"]
}

# 검색 예시 쿼리들
EXAMPLE_QUERIES = [
    "1층 평면도를 찾아주세요",
    "전용면적이 60제곱미터 이상인 세대 도면",
    "2023년 이후 설계된 입면도",
    "표가 포함된 상세도면들",
    "부산 지역의 공동주택 배치도",
    "완성도가 80점 이상인 도면들"
]

def get_attribute_info():
    """AttributeInfo 리스트 반환"""
    return ARCHITECTURAL_METADATA_FIELDS

def get_document_description():
    """문서 내용 설명 반환"""
    return DOCUMENT_CONTENT_DESCRIPTION

def validate_metadata(metadata_dict):
    """메타데이터 유효성 검증"""
    required_fields = ["drawing_type", "project_name"]
    
    for field in required_fields:
        if field not in metadata_dict or not metadata_dict[field]:
            return False, f"필수 필드 누락: {field}"
    
    # 타입 검증
    type_validations = {
        "floor_level": int,
        "design_year": int, 
        "exclusive_area": (int, float),
        "supply_area": (int, float),
        "site_area": (int, float),
        "has_tables": bool,
        "has_dimensions": bool,
        "material_count": int,
        "room_count": int,
        "completion_score": int
    }
    
    for field, expected_type in type_validations.items():
        if field in metadata_dict and metadata_dict[field] is not None:
            if not isinstance(metadata_dict[field], expected_type):
                return False, f"필드 {field}의 타입이 올바르지 않습니다. 기대: {expected_type}"
    
    return True, "유효함"

if __name__ == "__main__":
    # 설정 테스트
    print("🏗️ 건축 도면 Self-Query 설정")
    print(f"📊 메타데이터 필드 수: {len(ARCHITECTURAL_METADATA_FIELDS)}")
    print(f"📝 문서 설명: {DOCUMENT_CONTENT_DESCRIPTION}")
    print("\n🔍 지원하는 필드들:")
    for field in ARCHITECTURAL_METADATA_FIELDS:
        print(f"  - {field.name} ({field.type}): {field.description}")
