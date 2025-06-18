#!/usr/bin/env python3
"""
DWG/DXF 파일 LLM 기반 메타데이터 추출기
DWG 파서에서 추출한 구조적 데이터를 LLM을 통해 의미있는 메타데이터로 변환합니다.
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
    
    @trace_llm_call("dwg_content_analysis")
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
        "has_dimensions": true/false,
        "dimension_types": ["치수 유형들"],
        "measurement_units": "측정 단위"
    }},
    "annotations": {{
        "has_annotations": true/false,
        "annotation_types": ["주석 유형들"],
        "key_annotations": ["중요 주석 내용들"]
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
                logger.warning("내용 분석 JSON 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"내용 분석 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"도면 내용 분석 실패: {e}")
            return {}
    
    @trace_llm_call("dwg_architectural_features")
    def _extract_architectural_features(self, dwg_summary: str, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """건축적 특징 추출"""
        
        # 레이어 정보로 건축적 특징 추정
        layers = raw_metadata.get('layers', [])
        layer_names = [layer.get('name', '') for layer in layers]
        layer_summary = ", ".join(layer_names[:20])  # 상위 20개 레이어
        
        # 블록 정보로 반복 요소 파악
        blocks = raw_metadata.get('blocks', [])
        block_names = [block.get('name', '') for block in blocks]
        block_summary = ", ".join(block_names[:10])  # 상위 10개 블록
        
        prompt_template = """
다음은 건축 도면의 분석 정보입니다:

=== 도면 구조 ===
{dwg_summary}

=== 레이어 정보 ===
주요 레이어: {layer_summary}

=== 블록 정보 ===
주요 블록: {block_summary}

이 정보를 바탕으로 도면의 건축적 특징을 분석해주세요.

다음 형식으로 JSON 응답해주세요:
{{
    "architectural_features": {{
        "building_elements": ["벽체", "문", "창문", "계단", "기타"],
        "structural_elements": ["기둥", "보", "슬래브", "기타"],
        "spatial_features": ["공간적 특징들"],
        "design_patterns": ["설계 패턴들"]
    }},
    "construction_details": {{
        "detail_level": "상세도 수준 (개략, 기본, 상세)",
        "construction_phases": ["건설 단계들"],
        "material_indications": ["재료 표시들"]
    }},
    "compliance_and_standards": {{
        "drawing_standards": "도면 표준 (추정)",
        "code_compliance": ["적용 기준들"],
        "professional_stamps": "전문가 도장 여부"
    }}
}}
"""
        
        try:
            formatted_prompt = prompt_template.format(
                dwg_summary=dwg_summary,
                layer_summary=layer_summary,
                block_summary=block_summary
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
                logger.warning("건축적 특징 JSON 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"건축적 특징 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"건축적 특징 추출 실패: {e}")
            return {}
    
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
            
            return "\n".join(rag_content)
            
        except Exception as e:
            logger.error(f"RAG 콘텐츠 생성 실패: {e}")
            return ""

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
        
        try:
            # 프로젝트 처리기 초기화
            processor = DWGProjectProcessor(uploads_path)
            
            # 프로젝트 처리
            project_result = processor.process_project(project_name)
            
            if not project_result or project_result.get('status') != 'completed':
                logger.error(f"프로젝트 처리 실패: {project_name}")
                return {}
            
            # 각 파일별 LLM 메타데이터 추출
            enhanced_metadata = {}
            processed_files = project_result.get('processed_files', {})
            
            for file_name, file_data in processed_files.items():
                logger.info(f"LLM 메타데이터 추출 시작: {file_name}")
                
                try:
                    # 파일별 구조적 데이터 가져오기
                    raw_metadata = file_data.get('metadata', {})
                    dwg_summary = file_data.get('summary', '')
                    
                    # LLM 메타데이터 추출
                    file_metadata = self._extract_file_metadata(dwg_summary, raw_metadata, file_name)
                    
                    if file_metadata:
                        enhanced_metadata[file_name] = {
                            'file_info': file_data.get('file_info', {}),
                            'structural_metadata': raw_metadata,
                            'llm_metadata': file_metadata,
                            'rag_content': self.generate_rag_content(file_metadata),
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        logger.info(f"LLM 메타데이터 추출 완료: {file_name}")
                    else:
                        logger.warning(f"LLM 메타데이터 추출 실패: {file_name}")
                        
                except Exception as e:
                    logger.error(f"파일 메타데이터 추출 오류 ({file_name}): {e}")
            
            # 프로젝트 통합 메타데이터
            project_metadata = {
                'project_info': {
                    'project_name': project_name,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'dwg_project_llm_analysis',
                    'model_used': self.model_name,
                    'total_files': len(processed_files),
                    'successfully_processed': len(enhanced_metadata)
                },
                'processing_summary': project_result.get('processing_summary', {}),
                'files_metadata': enhanced_metadata,
                'project_analysis': self._analyze_project_patterns(enhanced_metadata)
            }
            
            # 프로젝트 메타데이터 저장
            self._save_project_metadata(project_name, project_metadata, uploads_path)
            
            return project_metadata
            
        except Exception as e:
            logger.error(f"프로젝트 메타데이터 추출 실패: {e}")
            return {}
    
    def _extract_file_metadata(self, dwg_summary: str, raw_metadata: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """단일 파일의 LLM 메타데이터 추출"""
        try:
            extracted_metadata = {}
            
            # 기본 도면 정보 추출
            basic_metadata = self._extract_basic_metadata(dwg_summary, raw_metadata)
            extracted_metadata.update(basic_metadata)
            
            # 도면 내용 분석
            content_analysis = self._analyze_drawing_content(dwg_summary, raw_metadata)
            extracted_metadata.update(content_analysis)
            
            # 건축적 특징 추출
            architectural_features = self._extract_architectural_features(dwg_summary, raw_metadata)
            extracted_metadata.update(architectural_features)
            
            # 기술적 메타데이터
            technical_metadata = self._extract_technical_metadata(raw_metadata)
            extracted_metadata.update(technical_metadata)
            
            # 파일별 정보 추가
            extracted_metadata['file_metadata'] = {
                'file_name': file_name,
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_method': 'dwg_llm_analysis',
                'model_used': self.model_name
            }
            
            return extracted_metadata
            
        except Exception as e:
            logger.error(f"파일 메타데이터 추출 실패 ({file_name}): {e}")
            return {}
    
    def _analyze_project_patterns(self, files_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """프로젝트 전체의 패턴 및 특징 분석"""
        try:
            if not files_metadata:
                return {}
            
            # 프로젝트 전체 통계
            total_files = len(files_metadata)
            drawing_types = {}
            disciplines = {}
            building_types = {}
            complexity_scores = []
            
            for file_name, file_data in files_metadata.items():
                llm_meta = file_data.get('llm_metadata', {})
                
                # 도면 유형 통계
                project_info = llm_meta.get('project_info', {})
                drawing_type = project_info.get('drawing_type', 'Unknown')
                drawing_types[drawing_type] = drawing_types.get(drawing_type, 0) + 1
                
                # 분야 통계
                discipline = project_info.get('discipline', 'Unknown')
                disciplines[discipline] = disciplines.get(discipline, 0) + 1
                
                # 건물 유형 통계  
                drawing_meta = llm_meta.get('drawing_metadata', {})
                building_type = drawing_meta.get('building_type', 'Unknown')
                building_types[building_type] = building_types.get(building_type, 0) + 1
                
                # 복잡도 점수
                tech_specs = llm_meta.get('complexity_metrics', {})
                complexity_score = tech_specs.get('complexity_score', 0)
                if complexity_score > 0:
                    complexity_scores.append(complexity_score)
            
            # 평균 복잡도 계산
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
            
            project_analysis = {
                'project_statistics': {
                    'total_files': total_files,
                    'drawing_types': drawing_types,
                    'disciplines': disciplines,
                    'building_types': building_types,
                    'average_complexity': round(avg_complexity, 2)
                },
                'project_characteristics': {
                    'dominant_drawing_type': max(drawing_types.items(), key=lambda x: x[1])[0] if drawing_types else 'Unknown',
                    'primary_discipline': max(disciplines.items(), key=lambda x: x[1])[0] if disciplines else 'Unknown',
                    'main_building_type': max(building_types.items(), key=lambda x: x[1])[0] if building_types else 'Unknown',
                    'complexity_level': self._categorize_complexity(avg_complexity)
                }
            }
            
            return project_analysis
            
        except Exception as e:
            logger.error(f"프로젝트 패턴 분석 실패: {e}")
            return {}
    
    def _categorize_complexity(self, score: float) -> str:
        """복잡도 점수를 카테고리로 변환"""
        if score < 2.0:
            return "단순"
        elif score < 5.0:
            return "보통"
        elif score < 8.0:
            return "복잡"
        else:
            return "매우복잡"
    
    def _save_project_metadata(self, project_name: str, metadata: Dict[str, Any], uploads_path: str = None):
        """프로젝트 메타데이터 저장"""
        try:
            if uploads_path is None:
                uploads_path = Path(__file__).parent.parent / "uploads"
            else:
                uploads_path = Path(uploads_path)
            
            # 프로젝트별 메타데이터 폴더 생성
            project_dir = uploads_path / project_name / 'processed_metadata'
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # 통합 메타데이터 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata_file = project_dir / f"project_llm_metadata_{timestamp}.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"프로젝트 메타데이터 저장: {metadata_file}")
            
            # RAG 콘텐츠 통합 파일 생성
            self._generate_project_rag_content(project_name, metadata, project_dir)
            
        except Exception as e:
            logger.error(f"프로젝트 메타데이터 저장 실패: {e}")
    
    def _generate_project_rag_content(self, project_name: str, metadata: Dict[str, Any], output_dir: Path):
        """프로젝트 전체 RAG 콘텐츠 생성"""
        try:
            rag_lines = []
            
            # 프로젝트 헤더
            rag_lines.append(f"=== 프로젝트: {project_name} ===")
            rag_lines.append(f"추출 시간: {metadata.get('project_info', {}).get('extraction_timestamp', '')}")
            
            # 프로젝트 특성
            project_analysis = metadata.get('project_analysis', {})
            project_chars = project_analysis.get('project_characteristics', {})
            
            if project_chars:
                rag_lines.append(f"\n=== 프로젝트 특성 ===")
                rag_lines.append(f"주요 도면 유형: {project_chars.get('dominant_drawing_type', '')}")
                rag_lines.append(f"주요 분야: {project_chars.get('primary_discipline', '')}")
                rag_lines.append(f"건물 유형: {project_chars.get('main_building_type', '')}")
                rag_lines.append(f"복잡도: {project_chars.get('complexity_level', '')}")
            
            # 파일별 내용
            files_metadata = metadata.get('files_metadata', {})
            if files_metadata:
                rag_lines.append(f"\n=== 도면 파일별 내용 ===")
                
                for file_name, file_data in files_metadata.items():
                    rag_content = file_data.get('rag_content', '')
                    if rag_content:
                        rag_lines.append(f"\n--- {file_name} ---")
                        rag_lines.append(rag_content)
            
            # RAG 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rag_file = output_dir / f"project_rag_content_{timestamp}.txt"
            
            with open(rag_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rag_lines))
            
            logger.info(f"프로젝트 RAG 콘텐츠 저장: {rag_file}")
            
        except Exception as e:
            logger.error(f"프로젝트 RAG 콘텐츠 생성 실패: {e}")
    
    def _analyze_xref_relationships(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """XREF 관계 분석"""
        xrefs = raw_metadata.get('xrefs', [])
        
        if not xrefs:
            return {}
        
        try:
            xref_blocks = [x for x in xrefs if x.get('type') == 'XREF_BLOCK']
            xref_inserts = [x for x in xrefs if x.get('type') == 'XREF_INSERT']
            
            xref_analysis = {
                'xref_relationships': {
                    'total_xrefs': len(xrefs),
                    'xref_blocks': len(xref_blocks),
                    'xref_inserts': len(xref_inserts),
                    'referenced_files': [x.get('filename', '') for x in xref_blocks if x.get('filename')],
                    'overlay_count': len([x for x in xref_blocks if x.get('is_overlay', False)]),
                    'attachment_count': len([x for x in xref_blocks if not x.get('is_overlay', False)])
                },
                'xref_details': {
                    'blocks': xref_blocks,
                    'inserts': xref_inserts
                }
            }
            
            logger.info(f"XREF 관계 분석 완료: {len(xrefs)}개 XREF 발견")
            return xref_analysis
            
        except Exception as e:
            logger.error(f"XREF 관계 분석 실패: {e}")
            return {}

    @trace_llm_call("dwg_basic_metadata")
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
    
    @trace_llm_call("dwg_content_analysis")
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
        "has_dimensions": true/false,
        "dimension_types": ["치수 유형들"],
        "measurement_units": "측정 단위"
    }},
    "annotations": {{
        "has_annotations": true/false,
        "annotation_types": ["주석 유형들"],
        "key_annotations": ["중요 주석 내용들"]
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
                logger.warning("내용 분석 JSON 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"내용 분석 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"도면 내용 분석 실패: {e}")
            return {}
    
    @trace_llm_call("dwg_architectural_features")
    def _extract_architectural_features(self, dwg_summary: str, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """건축적 특징 추출"""
        
        # 레이어 정보로 건축적 특징 추정
        layers = raw_metadata.get('layers', [])
        layer_names = [layer.get('name', '') for layer in layers]
        layer_summary = ", ".join(layer_names[:20])  # 상위 20개 레이어
        
        # 블록 정보로 반복 요소 파악
        blocks = raw_metadata.get('blocks', [])
        block_names = [block.get('name', '') for block in blocks]
        block_summary = ", ".join(block_names[:10])  # 상위 10개 블록
        
        prompt_template = """
다음은 건축 도면의 분석 정보입니다:

=== 도면 구조 ===
{dwg_summary}

=== 레이어 정보 ===
주요 레이어: {layer_summary}

=== 블록 정보 ===
주요 블록: {block_summary}

이 정보를 바탕으로 도면의 건축적 특징을 분석해주세요.

다음 형식으로 JSON 응답해주세요:
{{
    "architectural_features": {{
        "building_elements": ["벽체", "문", "창문", "계단", "기타"],
        "structural_elements": ["기둥", "보", "슬래브", "기타"],
        "spatial_features": ["공간적 특징들"],
        "design_patterns": ["설계 패턴들"]
    }},
    "construction_details": {{
        "detail_level": "상세도 수준 (개략, 기본, 상세)",
        "construction_phases": ["건설 단계들"],
        "material_indications": ["재료 표시들"]
    }},
    "compliance_and_standards": {{
        "drawing_standards": "도면 표준 (추정)",
        "code_compliance": ["적용 기준들"],
        "professional_stamps": "전문가 도장 여부"
    }}
}}
"""
        
        try:
            formatted_prompt = prompt_template.format(
                dwg_summary=dwg_summary,
                layer_summary=layer_summary,
                block_summary=block_summary
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
                logger.warning("건축적 특징 JSON 응답을 찾을 수 없습니다.")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"건축적 특징 JSON 파싱 오류: {e}")
            return {}
        except Exception as e:
            logger.error(f"건축적 특징 추출 실패: {e}")
            return {}
    
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
            
            return "\n".join(rag_content)
            
        except Exception as e:
            logger.error(f"RAG 콘텐츠 생성 실패: {e}")
            return ""

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
        
        try:
            # 프로젝트 처리기 초기화
            processor = DWGProjectProcessor(uploads_path)
            
            # 프로젝트 처리
            project_result = processor.process_project(project_name)
            
            if not project_result or project_result.get('status') != 'completed':
                logger.error(f"프로젝트 처리 실패: {project_name}")
                return {}
            
            # 각 파일별 LLM 메타데이터 추출
            enhanced_metadata = {}
            processed_files = project_result.get('processed_files', {})
            
            for file_name, file_data in processed_files.items():
                logger.info(f"LLM 메타데이터 추출 시작: {file_name}")
                
                try:
                    # 파일별 구조적 데이터 가져오기
                    raw_metadata = file_data.get('metadata', {})
                    dwg_summary = file_data.get('summary', '')
                    
                    # LLM 메타데이터 추출
                    file_metadata = self._extract_file_metadata(dwg_summary, raw_metadata, file_name)
                    
                    if file_metadata:
                        enhanced_metadata[file_name] = {
                            'file_info': file_data.get('file_info', {}),
                            'structural_metadata': raw_metadata,
                            'llm_metadata': file_metadata,
                            'rag_content': self.generate_rag_content(file_metadata),
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        logger.info(f"LLM 메타데이터 추출 완료: {file_name}")
                    else:
                        logger.warning(f"LLM 메타데이터 추출 실패: {file_name}")
                        
                except Exception as e:
                    logger.error(f"파일 메타데이터 추출 오류 ({file_name}): {e}")
            
            # 프로젝트 통합 메타데이터
            project_metadata = {
                'project_info': {
                    'project_name': project_name,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'dwg_project_llm_analysis',
                    'model_used': self.model_name,
                    'total_files': len(processed_files),
                    'successfully_processed': len(enhanced_metadata)
                },
                'processing_summary': project_result.get('processing_summary', {}),
                'files_metadata': enhanced_metadata,
                'project_analysis': self._analyze_project_patterns(enhanced_metadata)
            }
            
            # 프로젝트 메타데이터 저장
            self._save_project_metadata(project_name, project_metadata, uploads_path)
            
            return project_metadata
            
        except Exception as e:
            logger.error(f"프로젝트 메타데이터 추출 실패: {e}")
            return {}
    
    def _extract_file_metadata(self, dwg_summary: str, raw_metadata: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """단일 파일의 LLM 메타데이터 추출"""
        try:
            extracted_metadata = {}
            
            # 기본 도면 정보 추출
            basic_metadata = self._extract_basic_metadata(dwg_summary, raw_metadata)
            extracted_metadata.update(basic_metadata)
            
            # 도면 내용 분석
            content_analysis = self._analyze_drawing_content(dwg_summary, raw_metadata)
            extracted_metadata.update(content_analysis)
            
            # 건축적 특징 추출
            architectural_features = self._extract_architectural_features(dwg_summary, raw_metadata)
            extracted_metadata.update(architectural_features)
            
            # 기술적 메타데이터
            technical_metadata = self._extract_technical_metadata(raw_metadata)
            extracted_metadata.update(technical_metadata)
            
            # 파일별 정보 추가
            extracted_metadata['file_metadata'] = {
                'file_name': file_name,
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_method': 'dwg_llm_analysis',
                'model_used': self.model_name
            }
            
            return extracted_metadata
            
        except Exception as e:
            logger.error(f"파일 메타데이터 추출 실패 ({file_name}): {e}")
            return {}
    
    def _analyze_project_patterns(self, files_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """프로젝트 전체의 패턴 및 특징 분석"""
        try:
            if not files_metadata:
                return {}
            
            # 프로젝트 전체 통계
            total_files = len(files_metadata)
            drawing_types = {}
            disciplines = {}
            building_types = {}
            complexity_scores = []
            
            for file_name, file_data in files_metadata.items():
                llm_meta = file_data.get('llm_metadata', {})
                
                # 도면 유형 통계
                project_info = llm_meta.get('project_info', {})
                drawing_type = project_info.get('drawing_type', 'Unknown')
                drawing_types[drawing_type] = drawing_types.get(drawing_type, 0) + 1
                
                # 분야 통계
                discipline = project_info.get('discipline', 'Unknown')
                disciplines[discipline] = disciplines.get(discipline, 0) + 1
                
                # 건물 유형 통계  
                drawing_meta = llm_meta.get('drawing_metadata', {})
                building_type = drawing_meta.get('building_type', 'Unknown')
                building_types[building_type] = building_types.get(building_type, 0) + 1
                
                # 복잡도 점수
                tech_specs = llm_meta.get('complexity_metrics', {})
                complexity_score = tech_specs.get('complexity_score', 0)
                if complexity_score > 0:
                    complexity_scores.append(complexity_score)
            
            # 평균 복잡도 계산
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
            
            project_analysis = {
                'project_statistics': {
                    'total_files': total_files,
                    'drawing_types': drawing_types,
                    'disciplines': disciplines,
                    'building_types': building_types,
                    'average_complexity': round(avg_complexity, 2)
                },
                'project_characteristics': {
                    'dominant_drawing_type': max(drawing_types.items(), key=lambda x: x[1])[0] if drawing_types else 'Unknown',
                    'primary_discipline': max(disciplines.items(), key=lambda x: x[1])[0] if disciplines else 'Unknown',
                    'main_building_type': max(building_types.items(), key=lambda x: x[1])[0] if building_types else 'Unknown',
                    'complexity_level': self._categorize_complexity(avg_complexity)
                }
            }
            
            return project_analysis
            
        except Exception as e:
            logger.error(f"프로젝트 패턴 분석 실패: {e}")
            return {}
    
    def _categorize_complexity(self, score: float) -> str:
        """복잡도 점수를 카테고리로 변환"""
        if score < 2.0:
            return "단순"
        elif score < 5.0:
            return "보통"
        elif score < 8.0:
            return "복잡"
        else:
            return "매우복잡"
    
    def _save_project_metadata(self, project_name: str, metadata: Dict[str, Any], uploads_path: str = None):
        """프로젝트 메타데이터 저장"""
        try:
            if uploads_path is None:
                uploads_path = Path(__file__).parent.parent / "uploads"
            else:
                uploads_path = Path(uploads_path)
            
            # 프로젝트별 메타데이터 폴더 생성
            project_dir = uploads_path / project_name / 'processed_metadata'
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # 통합 메타데이터 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata_file = project_dir / f"project_llm_metadata_{timestamp}.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"프로젝트 메타데이터 저장: {metadata_file}")
            
            # RAG 콘텐츠 통합 파일 생성
            self._generate_project_rag_content(project_name, metadata, project_dir)
            
        except Exception as e:
            logger.error(f"프로젝트 메타데이터 저장 실패: {e}")
    
    def _generate_project_rag_content(self, project_name: str, metadata: Dict[str, Any], output_dir: Path):
        """프로젝트 전체 RAG 콘텐츠 생성"""
        try:
            rag_lines = []
            
            # 프로젝트 헤더
            rag_lines.append(f"=== 프로젝트: {project_name} ===")
            rag_lines.append(f"추출 시간: {metadata.get('project_info', {}).get('extraction_timestamp', '')}")
            
            # 프로젝트 특성
            project_analysis = metadata.get('project_analysis', {})
            project_chars = project_analysis.get('project_characteristics', {})
            
            if project_chars:
                rag_lines.append(f"\n=== 프로젝트 특성 ===")
                rag_lines.append(f"주요 도면 유형: {project_chars.get('dominant_drawing_type', '')}")
                rag_lines.append(f"주요 분야: {project_chars.get('primary_discipline', '')}")
                rag_lines.append(f"건물 유형: {project_chars.get('main_building_type', '')}")
                rag_lines.append(f"복잡도: {project_chars.get('complexity_level', '')}")
            
            # 파일별 내용
            files_metadata = metadata.get('files_metadata', {})
            if files_metadata:
                rag_lines.append(f"\n=== 도면 파일별 내용 ===")
                
                for file_name, file_data in files_metadata.items():
                    rag_content = file_data.get('rag_content', '')
                    if rag_content:
                        rag_lines.append(f"\n--- {file_name} ---")
                        rag_lines.append(rag_content)
            
            # RAG 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rag_file = output_dir / f"project_rag_content_{timestamp}.txt"
            
            with open(rag_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rag_lines))
            
            logger.info(f"프로젝트 RAG 콘텐츠 저장: {rag_file}")
            
        except Exception as e:
            logger.error(f"프로젝트 RAG 콘텐츠 생성 실패: {e}")

def main():
    """테스트용 메인 함수"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python dwg_metadata_extractor.py <command> [options]")
        print("명령어:")
        print("  extract_project <project_name>  - 프로젝트 메타데이터 추출")
        print("  extract_file <file_path>        - 단일 파일 메타데이터 추출")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "extract_project" and len(sys.argv) >= 3:
        # 프로젝트 메타데이터 추출
        project_name = sys.argv[2]
        extractor = DWGMetadataExtractor()
        
        print(f"=== 프로젝트 '{project_name}' 메타데이터 추출 시작 ===")
        metadata = extractor.extract_from_project(project_name)
        
        if metadata:
            project_info = metadata.get('project_info', {})
            print(f"추출 완료:")
            print(f"  - 총 파일: {project_info.get('total_files', 0)}개")
            print(f"  - 처리 성공: {project_info.get('successfully_processed', 0)}개")
            
            # 프로젝트 특성 출력
            project_analysis = metadata.get('project_analysis', {})
            project_chars = project_analysis.get('project_characteristics', {})
            
            if project_chars:
                print(f"\n프로젝트 특성:")
                print(f"  - 주요 도면 유형: {project_chars.get('dominant_drawing_type', 'Unknown')}")
                print(f"  - 주요 분야: {project_chars.get('primary_discipline', 'Unknown')}")
                print(f"  - 건물 유형: {project_chars.get('main_building_type', 'Unknown')}")
                print(f"  - 복잡도: {project_chars.get('complexity_level', 'Unknown')}")
        else:
            print("프로젝트 메타데이터 추출에 실패했습니다.")
            sys.exit(1)
    
    elif command == "extract_file" and len(sys.argv) >= 3:
        # 단일 파일 메타데이터 추출 (기존 기능)
        file_path = sys.argv[2]
        extractor = DWGMetadataExtractor()
        
        metadata = extractor.extract_from_dwg_file(file_path)
        
        if metadata:
            print("=== DWG 메타데이터 추출 결과 ===")
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # JSON 저장
            output_path = f"{Path(file_path).stem}_llm_metadata.json"
            extractor.save_metadata(metadata, output_path)
            print(f"\n메타데이터가 {output_path}에 저장되었습니다.")
            
            # RAG 콘텐츠 생성
            rag_content = extractor.generate_rag_content(metadata)
            print(f"\n=== RAG 콘텐츠 ===")
            print(rag_content)
            
        else:
            print("메타데이터 추출에 실패했습니다.")
            sys.exit(1)
    
    else:
        print("잘못된 명령어입니다. 사용법을 확인해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()
