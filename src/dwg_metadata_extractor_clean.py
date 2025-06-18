#!/usr/bin/env python3
"""
DWG/DXF 파일 LLM 기반 메타데이터 추출기 (정리된 버전)
DWG 파서에서 추출한 구조적 데이터를 LLM을 통해 의미있는 메타데이터로 변환합니다.
모든 주요 메서드에 LangSmith 추적이 일관되게 적용되었습니다.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from datetime import datetime

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    print("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama로 설치해주세요.")
    ChatOllama = None
    HAS_OLLAMA = False

# 절대 import 또는 sys.path 기반 import
try:
    from prompt_manager import get_prompt_manager
    from dwg_parser import DWGParser, DWGProjectProcessor
    from env_config import EnvironmentConfig, get_env_str
    from langsmith_integration import trace_llm_call, LangSmithTracker
except ImportError:
    import sys
    from pathlib import Path
    current_dir = str(Path(__file__).parent)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from prompt_manager import get_prompt_manager
    from dwg_parser import DWGParser, DWGProjectProcessor
    from env_config import EnvironmentConfig, get_env_str
    from langsmith_integration import trace_llm_call, LangSmithTracker

logger = logging.getLogger(__name__)

class DWGMetadataExtractor:
    """DWG/DXF 파일에서 LLM을 사용한 메타데이터 추출기"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        """
        Args:
            model_name: 사용할 Ollama 모델명 (None이면 .env에서 로드)
            base_url: Ollama 서버 주소 (기본값: http://localhost:11434)
        """
        self.env_config = EnvironmentConfig()
        self.model_name = model_name or self.env_config.model_config.model_name
        self.prompt_manager = get_prompt_manager()
        
        # Ollama 서버 설정
        self.base_url = base_url or get_env_str('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        self.llm = None
        self.dwg_parser = DWGParser()
        self._initialize_llm()
        
    def _initialize_llm(self):
        """LLM 초기화 - Ollama 서버와 연결"""
        if not HAS_OLLAMA:
            logger.error("langchain-ollama가 설치되지 않았습니다.")
            return
            
        try:
            # LangSmith 추적 설정
            self.langsmith_tracker = LangSmithTracker()
            
            # Ollama ChatOllama 연결
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.1,  # 메타데이터 추출은 일관성이 중요
                num_predict=2048,  # DWG 분석용으로 충분한 토큰
                timeout=120,  # DWG 분석은 시간이 걸릴 수 있음
            )
            
            logger.info(f"DWG 메타데이터 추출용 LLM 모델 '{self.model_name}' 초기화 완료")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            self.llm = None

    @trace_llm_call("dwg_file_metadata_extraction", "main_extraction")
    def extract_from_dwg_file(self, dwg_file_path: str, project_base_path: str = None) -> Dict[str, Any]:
        """
        DWG/DXF 파일에서 직접 메타데이터 추출 (XREF 포함)
        
        Args:
            dwg_file_path: DWG/DXF 파일 경로
            project_base_path: 프로젝트 기본 경로 (XREF 파일 검색용)
            
        Returns:
            Dict: 추출된 메타데이터
        """
        if not self.llm:
            logger.error("LLM이 초기화되지 않았습니다.")
            return {}
        
        # 1. DWG 파서로 구조적 데이터 추출 (XREF 포함)
        logger.info(f"DWG 파일 파싱 시작: {dwg_file_path}")
        
        # XREF와 함께 로드
        if project_base_path:
            if not self.dwg_parser.load_file_with_xref(dwg_file_path, project_base_path):
                logger.error(f"DWG 파일 로드 실패: {dwg_file_path}")
                return {}
        else:
            if not self.dwg_parser.load_file(dwg_file_path):
                logger.error(f"DWG 파일 로드 실패: {dwg_file_path}")
                return {}
        
        # 2. 구조적 데이터를 LLM이 읽기 쉬운 형태로 변환
        dwg_summary = self.dwg_parser.generate_llm_readable_summary()
        raw_metadata = self.dwg_parser.extract_all_metadata()
        
        # 3. LLM을 통한 메타데이터 추출
        logger.info("LLM을 통한 메타데이터 추출 시작")
        
        extracted_metadata = {}
        
        try:
            # 기본 도면 정보 추출
            basic_metadata = self._extract_basic_metadata(dwg_summary, raw_metadata)
            extracted_metadata.update(basic_metadata)
            
            # 도면 내용 분석
            content_analysis = self._analyze_drawing_content(dwg_summary, raw_metadata)
            extracted_metadata.update(content_analysis)
            
            # 건축적 특징 추출
            architectural_features = self._extract_architectural_features(dwg_summary, raw_metadata)
            extracted_metadata.update(architectural_features)
            
            # XREF 관계 분석
            xref_analysis = self._analyze_xref_relationships(raw_metadata)
            if xref_analysis:
                extracted_metadata.update(xref_analysis)
            
            # 기술적 메타데이터
            technical_metadata = self._extract_technical_metadata(raw_metadata)
            extracted_metadata.update(technical_metadata)
            
            # 추출 시간 정보 추가
            extracted_metadata['extraction_info'] = {
                'extraction_timestamp': datetime.now().isoformat(),
                'source_file': dwg_file_path,
                'extraction_method': 'dwg_llm_analysis_with_xref',
                'model_used': self.model_name,
                'xref_processed': bool(raw_metadata.get('xrefs'))
            }
            
            logger.info("DWG 메타데이터 추출 완료")
            return extracted_metadata
            
        except Exception as e:
            logger.error(f"메타데이터 추출 중 오류: {e}")
            return {}
    
    @trace_llm_call("dwg_basic_metadata_extraction", "llm")
    def _extract_basic_metadata(self, dwg_summary: str, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """기본 도면 메타데이터 추출"""
        
        prompt_template = """
다음은 CAD 도면에서 추출한 구조적 정보입니다. 이 정보를 바탕으로 도면의 기본 메타데이터를 추출해주세요.

{dwg_summary}

다음 형식으로 JSON 응답해주세요:
{{
    "project_info": {{
        "project_name": "프로젝트명 (추정)",
        "drawing_type": "도면 유형 (평면도, 입면도, 단면도, 상세도 등)",
        "drawing_purpose": "도면 목적",
        "scale": "축척 정보 (추정)",
        "discipline": "분야 (건축, 구조, 설비 등)"
    }},
    "drawing_metadata": {{
        "title": "도면 제목",
        "description": "도면 설명",
        "keywords": ["키워드1", "키워드2", "키워드3"],
        "building_type": "건물 유형 (주거, 상업, 공공 등)",
        "complexity_level": "복잡도 (단순, 보통, 복잡)"
    }}
}}
"""
        
        try:
            formatted_prompt = prompt_template.format(dwg_summary=dwg_summary)
            response = self.llm.invoke(formatted_prompt)
            
            # JSON 파싱
            response_text = response.content.strip()
            # JSON 블록에서 실제 JSON 부분만 추출
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info("기본 메타데이터 추출 완료")
                return result
            else:
                logger.warning("JSON 형식 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"기본 메타데이터 추출 실패: {e}")
            return {}
    
    @trace_llm_call("dwg_content_analysis", "llm")
    def _analyze_drawing_content(self, dwg_summary: str, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """도면 내용 분석"""
        
        # 텍스트 엔티티 정보 구성
        text_entities = raw_metadata.get('text_entities', [])
        text_content = []
        for text in text_entities[:50]:  # 상위 50개만 분석
            if text.get('text', '').strip():
                text_content.append(f"[{text.get('type', 'TEXT')}] {text.get('text', '').strip()}")
        
        text_summary = "\n".join(text_content) if text_content else "텍스트 정보 없음"
        
        prompt_template = """
다음은 CAD 도면의 내용 분석을 위한 정보입니다:

=== 도면 구조 정보 ===
{dwg_summary}

=== 도면 내 텍스트 내용 ===
{text_summary}

이 정보를 바탕으로 도면의 내용을 분석해주세요.

다음 형식으로 JSON 응답해주세요:
{{
    "content_analysis": {{
        "main_elements": ["주요 구성 요소1", "주요 구성 요소2"],
        "spatial_organization": "공간 구성 방식",
        "functional_areas": ["기능적 영역1", "기능적 영역2"],
        "technical_elements": ["기술적 요소1", "기술적 요소2"]
    }},
    "dimensions_and_measurements": {{
        "has_dimensions": true,
        "dimension_types": ["치수 유형들"],
        "measurement_units": "측정 단위"
    }},
    "annotations": {{
        "has_annotations": true,
        "annotation_types": ["주석 유형들"],
        "key_notes": ["주요 노트1", "주요 노트2"]
    }}
}}
"""
        
        try:
            formatted_prompt = prompt_template.format(
                dwg_summary=dwg_summary,
                text_summary=text_summary
            )
            response = self.llm.invoke(formatted_prompt)
            
            # JSON 파싱
            response_text = response.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info("도면 내용 분석 완료")
                return result
            else:
                logger.warning("JSON 형식 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"내용 분석 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"도면 내용 분석 실패: {e}")
            return {}
    
    @trace_llm_call("dwg_architectural_features", "llm")
    def _extract_architectural_features(self, dwg_summary: str, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """건축적 특징 추출"""
        
        # 블록과 레이어 정보로 건축 요소 추정
        blocks = raw_metadata.get('blocks', [])
        layers = raw_metadata.get('layers', [])
        
        block_info = []
        for block in blocks[:20]:  # 상위 20개 블록
            if block.get('name', ''):
                block_info.append(f"블록: {block.get('name', '')} (삽입수: {block.get('insert_count', 0)})")
        
        layer_info = []
        for layer in layers[:30]:  # 상위 30개 레이어
            if layer.get('name', ''):
                layer_info.append(f"레이어: {layer.get('name', '')}")
        
        architectural_context = f"""
=== 블록 정보 ===
{chr(10).join(block_info) if block_info else "블록 정보 없음"}

=== 레이어 정보 ===
{chr(10).join(layer_info) if layer_info else "레이어 정보 없음"}
"""
        
        prompt_template = """
다음은 CAD 도면의 건축적 특징 분석을 위한 정보입니다:

=== 도면 구조 정보 ===
{dwg_summary}

{architectural_context}

이 정보를 바탕으로 건축적 특징을 분석해주세요.

다음 형식으로 JSON 응답해주세요:
{{
    "architectural_features": {{
        "building_elements": ["벽", "문", "창", "계단", "기둥"],
        "structural_systems": ["구조 시스템1", "구조 시스템2"],
        "spatial_characteristics": {{
            "room_types": ["공간 유형1", "공간 유형2"],
            "circulation_patterns": "동선 패턴",
            "spatial_relationships": "공간 관계"
        }},
        "design_style": "건축 양식",
        "building_scale": "건물 규모 (소형, 중형, 대형)"
    }},
    "construction_elements": {{
        "materials_indicated": ["재료1", "재료2"],
        "construction_details": ["시공 디테일1", "시공 디테일2"],
        "technical_specifications": ["기술 사양1", "기술 사양2"]
    }}
}}
"""
        
        try:
            formatted_prompt = prompt_template.format(
                dwg_summary=dwg_summary,
                architectural_context=architectural_context
            )
            response = self.llm.invoke(formatted_prompt)
            
            # JSON 파싱
            response_text = response.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info("건축적 특징 추출 완료")
                return result
            else:
                logger.warning("JSON 형식 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"건축적 특징 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"건축적 특징 추출 실패: {e}")
            return {}
    
    @trace_llm_call("dwg_xref_analysis", "metadata_processing")
    def _analyze_xref_relationships(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """XREF 관계 분석"""
        
        xrefs = raw_metadata.get('xrefs', [])
        if not xrefs:
            return {}
        
        try:
            xref_analysis = {
                'xref_relationships': {
                    'total_xrefs': len(xrefs),
                    'xref_files': [],
                    'dependency_chain': [],
                    'external_references': []
                }
            }
            
            for xref in xrefs:
                xref_info = {
                    'name': xref.get('name', ''),
                    'path': xref.get('path', ''),
                    'status': xref.get('status', 'unknown'),
                    'is_resolved': xref.get('is_resolved', False),
                    'includes_geometry': xref.get('includes_geometry', False)
                }
                
                xref_analysis['xref_relationships']['xref_files'].append(xref_info)
                
                if xref_info['is_resolved']:
                    xref_analysis['xref_relationships']['dependency_chain'].append(xref_info['name'])
                else:
                    xref_analysis['xref_relationships']['external_references'].append(xref_info['name'])
            
            logger.info(f"XREF 관계 분석 완료: {len(xrefs)}개 참조")
            return xref_analysis
            
        except Exception as e:
            logger.error(f"XREF 관계 분석 실패: {e}")
            return {}
    
    @trace_llm_call("dwg_technical_metadata", "metadata_processing")
    def _extract_technical_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """기술적 메타데이터 추출 (LLM 없이 직접 처리)"""
        
        try:
            basic_info = raw_metadata.get('basic_info', {})
            statistics = raw_metadata.get('statistics', {})
            layers = raw_metadata.get('layers', [])
            
            technical_metadata = {
                'technical_specifications': {
                    'dxf_version': basic_info.get('dxf_version', ''),
                    'file_format': Path(basic_info.get('file_path', '')).suffix.upper().replace('.', ''),
                    'units': basic_info.get('units', ''),
                    'coordinate_system': 'WCS',  # 일반적으로 World Coordinate System
                    'drawing_limits': basic_info.get('drawing_limits', {}),
                    'drawing_extents': basic_info.get('drawing_extents', {})
                },
                'complexity_metrics': {
                    'total_entities': statistics.get('total_entities', 0),
                    'entity_diversity': len(statistics.get('entity_types', {})),
                    'layer_count': statistics.get('layer_count', 0),
                    'block_count': statistics.get('block_count', 0),
                    'layout_count': statistics.get('layout_count', 0)
                },
                'layer_analysis': {
                    'total_layers': len(layers),
                    'active_layers': len([l for l in layers if not l.get('is_off', False)]),
                    'locked_layers': len([l for l in layers if l.get('is_locked', False)]),
                    'frozen_layers': len([l for l in layers if l.get('is_frozen', False)])
                }
            }
            
            # 복잡도 점수 계산
            complexity_score = self._calculate_complexity_score(statistics)
            technical_metadata['complexity_metrics']['complexity_score'] = complexity_score
            
            return technical_metadata
            
        except Exception as e:
            logger.error(f"기술적 메타데이터 추출 실패: {e}")
            return {}
    
    def _calculate_complexity_score(self, statistics: Dict[str, Any]) -> float:
        """도면 복잡도 점수 계산"""
        try:
            total_entities = statistics.get('total_entities', 0)
            entity_types = len(statistics.get('entity_types', {}))
            layer_count = statistics.get('layer_count', 0)
            block_count = statistics.get('block_count', 0)
            
            # 가중치를 적용한 복잡도 점수 계산
            complexity_score = (
                total_entities * 0.001 +  # 엔티티 수
                entity_types * 0.1 +      # 엔티티 유형 다양성
                layer_count * 0.05 +      # 레이어 수
                block_count * 0.02        # 블록 수
            )
            
            # 0-10 범위로 정규화
            normalized_score = min(complexity_score, 10.0)
            
            return round(normalized_score, 2)
            
        except Exception as e:
            logger.error(f"복잡도 점수 계산 실패: {e}")
            return 0.0

    @trace_llm_call("dwg_project_extraction", "main_extraction")
    def extract_from_project(self, project_name: str, uploads_path: str = None) -> Dict[str, Any]:
        """
        프로젝트 단위로 모든 DWG 파일에서 메타데이터 추출
        
        Args:
            project_name: 프로젝트명
            uploads_path: uploads 폴더 경로 (선택사항)
            
        Returns:
            Dict: 프로젝트 전체 메타데이터
        """
        if not self.llm:
            logger.error("LLM이 초기화되지 않았습니다.")
            return {}
        
        # 기본 uploads 경로 설정
        if uploads_path is None:
            uploads_path = str(Path(__file__).parent.parent / 'uploads')
        
        project_path = Path(uploads_path) / project_name
        if not project_path.exists():
            logger.error(f"프로젝트 경로를 찾을 수 없습니다: {project_path}")
            return {}
        
        # DWG 프로젝트 프로세서 초기화
        processor = DWGProjectProcessor()
        dwg_files = processor._find_dwg_files_in_project(str(project_path))
        
        if not dwg_files:
            logger.warning(f"프로젝트에서 DWG/DXF 파일을 찾을 수 없습니다: {project_name}")
            return {}
        
        logger.info(f"프로젝트 '{project_name}'에서 {len(dwg_files)}개 DWG/DXF 파일 발견")
        
        # 각 파일별 메타데이터 추출
        files_metadata = {}
        
        for dwg_file in dwg_files:
            logger.info(f"DWG 파일 처리 중: {dwg_file}")
            
            try:
                # 파일별 메타데이터 추출 (XREF 지원)
                file_metadata = self.extract_from_dwg_file(dwg_file, str(project_path))
                
                if file_metadata:
                    # 파일 상대 경로를 키로 사용
                    relative_path = str(Path(dwg_file).relative_to(project_path))
                    files_metadata[relative_path] = file_metadata
                    
            except Exception as e:
                logger.error(f"DWG 파일 처리 실패 {dwg_file}: {e}")
                continue
        
        if not files_metadata:
            logger.error(f"프로젝트에서 처리된 DWG 파일이 없습니다: {project_name}")
            return {}
        
        # 프로젝트 전체 패턴 분석
        project_patterns = self._analyze_project_patterns(files_metadata)
        
        # 프로젝트 메타데이터 구성
        project_metadata = {
            'project_info': {
                'project_name': project_name,
                'total_dwg_files': len(files_metadata),
                'analysis_timestamp': datetime.now().isoformat(),
                'extraction_method': 'project_batch_analysis'
            },
            'project_patterns': project_patterns,
            'files_metadata': files_metadata
        }
        
        logger.info(f"프로젝트 '{project_name}' 메타데이터 추출 완료: {len(files_metadata)}개 파일")
        return project_metadata

    @trace_llm_call("dwg_project_patterns", "llm")
    def _analyze_project_patterns(self, files_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """프로젝트 전체 패턴 분석"""
        
        if not files_metadata:
            return {}
        
        # 프로젝트 요약 정보 구성
        project_summary = []
        drawing_types = []
        building_types = []
        disciplines = []
        
        for file_path, metadata in files_metadata.items():
            project_info = metadata.get('project_info', {})
            drawing_metadata = metadata.get('drawing_metadata', {})
            
            if project_info.get('drawing_type'):
                drawing_types.append(project_info['drawing_type'])
            if drawing_metadata.get('building_type'):
                building_types.append(drawing_metadata['building_type'])
            if project_info.get('discipline'):
                disciplines.append(project_info['discipline'])
            
            # 파일별 요약
            file_summary = f"파일: {file_path}"
            if project_info.get('drawing_type'):
                file_summary += f" | 유형: {project_info['drawing_type']}"
            if drawing_metadata.get('title'):
                file_summary += f" | 제목: {drawing_metadata['title']}"
            
            project_summary.append(file_summary)
        
        summary_text = "\n".join(project_summary[:20])  # 상위 20개 파일만
        
        prompt_template = """
다음은 건축 프로젝트의 DWG 파일들에서 추출한 정보입니다:

{summary_text}

이 정보를 바탕으로 프로젝트 전체의 패턴과 특성을 분석해주세요.

다음 형식으로 JSON 응답해주세요:
{{
    "project_characteristics": {{
        "primary_building_type": "주 건물 유형",
        "project_scale": "프로젝트 규모",
        "design_phase": "설계 단계",
        "architectural_style": "건축 양식"
    }},
    "drawing_organization": {{
        "drawing_categories": ["도면 분류1", "도면 분류2"],
        "discipline_coverage": ["분야1", "분야2"],
        "completeness_level": "완성도 수준"
    }},
    "technical_summary": {{
        "complexity_overview": "복잡도 개요",
        "standardization_level": "표준화 수준",
        "coordination_quality": "조정 품질"
    }}
}}
"""
        
        try:
            formatted_prompt = prompt_template.format(summary_text=summary_text)
            response = self.llm.invoke(formatted_prompt)
            
            # JSON 파싱
            response_text = response.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # 통계 정보 추가
                result['statistics'] = {
                    'total_files': len(files_metadata),
                    'drawing_type_distribution': self._count_occurrences(drawing_types),
                    'building_type_distribution': self._count_occurrences(building_types),
                    'discipline_distribution': self._count_occurrences(disciplines)
                }
                
                logger.info("프로젝트 패턴 분석 완료")
                return result
            else:
                logger.warning("JSON 형식 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"프로젝트 패턴 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"프로젝트 패턴 분석 실패: {e}")
            return {}
    
    def _count_occurrences(self, items: List[str]) -> Dict[str, int]:
        """항목별 발생 횟수 계산"""
        counts = {}
        for item in items:
            if item:
                counts[item] = counts.get(item, 0) + 1
        return counts

    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> bool:
        """추출된 메타데이터를 JSON 파일로 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"DWG 메타데이터를 저장했습니다: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
            return False

    @trace_llm_call("dwg_rag_content_generation", "content_processing")
    def generate_rag_content(self, metadata: Dict[str, Any]) -> str:
        """RAG 시스템용 콘텐츠 생성"""
        try:
            rag_content = []
            
            # 프로젝트 정보
            project_info = metadata.get('project_info', {})
            if project_info:
                rag_content.append(f"프로젝트: {project_info.get('project_name', 'Unknown')}")
                rag_content.append(f"도면 유형: {project_info.get('drawing_type', 'Unknown')}")
                rag_content.append(f"분야: {project_info.get('discipline', 'Unknown')}")
            
            # 도면 메타데이터
            drawing_metadata = metadata.get('drawing_metadata', {})
            if drawing_metadata:
                rag_content.append(f"제목: {drawing_metadata.get('title', 'Unknown')}")
                rag_content.append(f"설명: {drawing_metadata.get('description', 'Unknown')}")
                keywords = drawing_metadata.get('keywords', [])
                if keywords:
                    rag_content.append(f"키워드: {', '.join(keywords)}")
            
            # 내용 분석
            content_analysis = metadata.get('content_analysis', {})
            if content_analysis:
                main_elements = content_analysis.get('main_elements', [])
                if main_elements:
                    rag_content.append(f"주요 구성 요소: {', '.join(main_elements)}")
                
                functional_areas = content_analysis.get('functional_areas', [])
                if functional_areas:
                    rag_content.append(f"기능적 영역: {', '.join(functional_areas)}")
            
            # 건축적 특징
            arch_features = metadata.get('architectural_features', {})
            if arch_features:
                building_elements = arch_features.get('building_elements', [])
                if building_elements:
                    rag_content.append(f"건축 요소: {', '.join(building_elements)}")
            
            # 기술적 사양
            tech_specs = metadata.get('technical_specifications', {})
            if tech_specs:
                rag_content.append(f"파일 형식: {tech_specs.get('file_format', 'Unknown')}")
                rag_content.append(f"단위: {tech_specs.get('units', 'Unknown')}")
            
            # XREF 정보
            xref_relationships = metadata.get('xref_relationships', {})
            if xref_relationships:
                total_xrefs = xref_relationships.get('total_xrefs', 0)
                if total_xrefs > 0:
                    rag_content.append(f"외부 참조: {total_xrefs}개 파일")
            
            return "\n".join(rag_content)
            
        except Exception as e:
            logger.error(f"RAG 콘텐츠 생성 실패: {e}")
            return ""

# 주요 함수들을 모듈 레벨에서 접근 가능하게 만들기
def create_dwg_extractor(model_name: str = None, base_url: str = None) -> DWGMetadataExtractor:
    """DWG 메타데이터 추출기 생성"""
    return DWGMetadataExtractor(model_name=model_name, base_url=base_url)

def extract_dwg_metadata(dwg_file_path: str, project_base_path: str = None, 
                        model_name: str = None) -> Dict[str, Any]:
    """DWG 파일에서 메타데이터 추출 (단축함수)"""
    extractor = create_dwg_extractor(model_name=model_name)
    return extractor.extract_from_dwg_file(dwg_file_path, project_base_path)

def extract_project_dwg_metadata(project_name: str, uploads_path: str = None,
                                model_name: str = None) -> Dict[str, Any]:
    """프로젝트의 모든 DWG 파일에서 메타데이터 추출 (단축함수)"""
    extractor = create_dwg_extractor(model_name=model_name)
    return extractor.extract_from_project(project_name, uploads_path)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 간단한 테스트
    extractor = create_dwg_extractor()
    
    # 예시: 단일 파일 분석
    # metadata = extractor.extract_from_dwg_file("/path/to/dwg/file.dwg")
    
    # 예시: 프로젝트 전체 분석  
    # project_metadata = extractor.extract_from_project("01_행복도시_6-3생활권M3BL_실시설계도면2차_건축도면")
    
    print("DWG 메타데이터 추출기 초기화 완료")
