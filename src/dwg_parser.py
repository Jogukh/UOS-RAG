#!/usr/bin/env python3
"""
DWG/DXF 파일 파서
ezdxf 라이브러리를 사용하여 CAD 파일에서 구조적 정보와 메타데이터를 추출합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import glob
import os

try:
    import ezdxf
    from ezdxf import recover, xref
    from ezdxf.addons import odafc
    HAS_EZDXF = True
    HAS_XREF = hasattr(ezdxf, 'xref')  # ezdxf 1.1+ 에서 XREF 지원
except ImportError:
    print("ezdxf가 설치되지 않았습니다. pip install ezdxf로 설치해주세요.")
    HAS_EZDXF = False
    HAS_XREF = False

logger = logging.getLogger(__name__)

class DWGParser:
    """
    개별 DWG/DXF 파일 파서
    단일 파일에서 구조적 정보와 메타데이터를 추출합니다.
    """
    
    def __init__(self):
        """DWG 파서 초기화"""
        if not HAS_EZDXF:
            raise ImportError("ezdxf 라이브러리가 필요합니다.")
        
        self.doc = None
        self.file_path = None
        self.xref_loaded = False
        
    def load_file(self, file_path: str) -> bool:
        """
        DWG/DXF 파일 로드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            로드 성공 여부
        """
        self.file_path = file_path
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return False
        
        try:
            # DXF 파일은 직접 로드
            if file_path_obj.suffix.lower() == '.dxf':
                self.doc = ezdxf.readfile(file_path)
                logger.info(f"DXF 파일 로드 성공: {file_path_obj.name}")
                return True
            
            # DWG 파일은 ODA File Converter 필요
            elif file_path_obj.suffix.lower() == '.dwg':
                try:
                    # ODA File Converter로 로드
                    self.doc = odafc.readfile(file_path)
                    logger.info(f"DWG 파일 로드 성공: {file_path_obj.name}")
                    return True
                except Exception as e:
                    logger.error(f"DWG 파일 로드 실패: {e}")
                    return False
            
            else:
                logger.error(f"지원하지 않는 파일 형식: {file_path_obj.suffix}")
                return False
                
        except Exception as e:
            logger.error(f"파일 로드 실패: {e}")
            return False
    
    def load_file_with_xref(self, file_path: str, project_base_path: str = None) -> bool:
        """
        XREF와 함께 DWG/DXF 파일 로드
        
        Args:
            file_path: 메인 파일 경로
            project_base_path: 프로젝트 기본 경로 (XREF 검색용)
            
        Returns:
            로드 성공 여부
        """
        # 먼저 메인 파일 로드
        if not self.load_file(file_path):
            return False
        
        # XREF 처리
        if HAS_XREF and self.doc:
            try:
                # XREF 정보 확인
                xrefs = []
                for block_record in self.doc.block_records:
                    if hasattr(block_record, 'is_xref') and block_record.is_xref:
                        xrefs.append({
                            'name': block_record.dxf.name,
                            'filename': getattr(block_record.dxf, 'xref_path', 'Unknown')
                        })
                
                if xrefs:
                    logger.info(f"XREF 참조 {len(xrefs)}개 발견")
                    self.xref_loaded = True
                
            except Exception as e:
                logger.warning(f"XREF 처리 중 오류: {e}")
        
        return True
    
    def extract_basic_info(self) -> Dict[str, Any]:
        """기본 파일 정보 추출"""
        if not self.doc:
            return {}
        
        try:
            basic_info = {
                'file_path': str(self.file_path),
                'dxf_version': self.doc.dxfversion,
                'units': self.doc.header.get('$INSUNITS', 0),
                'drawing_limits': {
                    'min': list(self.doc.header.get('$LIMMIN', (0, 0))),
                    'max': list(self.doc.header.get('$LIMMAX', (0, 0)))
                },
                'drawing_extents': {
                    'min': list(self.doc.header.get('$EXTMIN', (0, 0, 0))),
                    'max': list(self.doc.header.get('$EXTMAX', (0, 0, 0)))
                }
            }
            
            return basic_info
            
        except Exception as e:
            logger.error(f"기본 정보 추출 실패: {e}")
            return {}
    
    def extract_all_metadata(self) -> Dict[str, Any]:
        """전체 메타데이터 추출"""
        if not self.doc:
            return {}
        
        try:
            metadata = {
                'basic_info': self.extract_basic_info(),
                'statistics': self._extract_statistics(),
                'layers': self._extract_layers(),
                'blocks': self._extract_blocks(),
                'text_entities': self._extract_text_entities(),
                'xrefs': self._extract_xrefs() if self.xref_loaded else []
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {e}")
            return {}
    
    def _extract_statistics(self) -> Dict[str, Any]:
        """통계 정보 추출"""
        try:
            msp = self.doc.modelspace()
            entities = list(msp)
            
            # 엔티티 유형별 통계
            entity_types = {}
            for entity in entities:
                entity_type = entity.dxftype()
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            statistics = {
                'total_entities': len(entities),
                'entity_types': entity_types,
                'layer_count': len(self.doc.layers),
                'block_count': len(self.doc.blocks),
                'layout_count': len(self.doc.layouts)
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"통계 정보 추출 실패: {e}")
            return {}
    
    def _extract_layers(self) -> List[Dict[str, Any]]:
        """레이어 정보 추출"""
        try:
            layers = []
            for layer in self.doc.layers:
                layer_info = {
                    'name': layer.dxf.name,
                    'color': getattr(layer.dxf, 'color', 7),
                    'linetype': getattr(layer.dxf, 'linetype', 'CONTINUOUS'),
                    'is_off': layer.is_off(),
                    'is_locked': layer.is_locked(),
                    'is_frozen': layer.is_frozen()
                }
                layers.append(layer_info)
            
            return layers
            
        except Exception as e:
            logger.error(f"레이어 정보 추출 실패: {e}")
            return []
    
    def _extract_blocks(self) -> List[Dict[str, Any]]:
        """블록 정보 추출"""
        try:
            blocks = []
            for block in self.doc.blocks:
                if not block.name.startswith('*'):  # 시스템 블록 제외
                    block_info = {
                        'name': block.name,
                        'entities_count': len(block),
                        'is_xref': getattr(block, 'is_xref', False)
                    }
                    blocks.append(block_info)
            
            return blocks
            
        except Exception as e:
            logger.error(f"블록 정보 추출 실패: {e}")
            return []
    
    def _extract_text_entities(self) -> List[Dict[str, Any]]:
        """텍스트 엔티티 정보 추출"""
        try:
            text_entities = []
            msp = self.doc.modelspace()
            
            for entity in msp.query('TEXT MTEXT'):
                text_info = {
                    'type': entity.dxftype(),
                    'text': getattr(entity.dxf, 'text', ''),
                    'layer': getattr(entity.dxf, 'layer', '0'),
                    'height': getattr(entity.dxf, 'height', 0)
                }
                
                if hasattr(entity.dxf, 'insert'):
                    text_info['position'] = list(entity.dxf.insert)
                
                text_entities.append(text_info)
            
            return text_entities[:100]  # 상위 100개만 반환
            
        except Exception as e:
            logger.error(f"텍스트 엔티티 추출 실패: {e}")
            return []
    
    def _extract_xrefs(self) -> List[Dict[str, Any]]:
        """XREF 정보 추출"""
        try:
            xrefs = []
            for block_record in self.doc.block_records:
                if hasattr(block_record, 'is_xref') and block_record.is_xref:
                    xref_info = {
                        'name': block_record.dxf.name,
                        'path': getattr(block_record.dxf, 'xref_path', 'Unknown'),
                        'status': 'resolved',
                        'is_resolved': True,
                        'includes_geometry': True
                    }
                    xrefs.append(xref_info)
            
            return xrefs
            
        except Exception as e:
            logger.error(f"XREF 정보 추출 실패: {e}")
            return []
    
    def generate_llm_readable_summary(self) -> str:
        """LLM이 읽기 쉬운 형태의 요약 생성"""
        if not self.doc:
            return "파일이 로드되지 않았습니다."
        
        try:
            metadata = self.extract_all_metadata()
            
            summary_parts = []
            
            # 기본 정보
            basic_info = metadata.get('basic_info', {})
            if basic_info:
                summary_parts.append(f"파일: {Path(basic_info.get('file_path', '')).name}")
                summary_parts.append(f"DXF 버전: {basic_info.get('dxf_version', 'Unknown')}")
                summary_parts.append(f"단위: {basic_info.get('units', 'Unknown')}")
            
            # 통계 정보
            stats = metadata.get('statistics', {})
            if stats:
                summary_parts.append(f"총 엔티티: {stats.get('total_entities', 0)}개")
                summary_parts.append(f"레이어: {stats.get('layer_count', 0)}개")
                summary_parts.append(f"블록: {stats.get('block_count', 0)}개")
                
                entity_types = stats.get('entity_types', {})
                if entity_types:
                    top_entities = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]
                    entities_str = ", ".join([f"{k}({v})" for k, v in top_entities])
                    summary_parts.append(f"주요 엔티티: {entities_str}")
            
            # 텍스트 정보
            texts = metadata.get('text_entities', [])
            if texts:
                text_samples = [t.get('text', '').strip() for t in texts[:5] if t.get('text', '').strip()]
                if text_samples:
                    summary_parts.append(f"텍스트 샘플: {', '.join(text_samples)}")
            
            # XREF 정보
            xrefs = metadata.get('xrefs', [])
            if xrefs:
                summary_parts.append(f"외부 참조: {len(xrefs)}개")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return f"요약 생성 실패: {e}"

class DWGProjectProcessor:
    """
    프로젝트 단위 DWG/DXF 파일 처리기
    uploads 폴더의 프로젝트별로 하위 폴더 내 모든 CAD 파일을 처리합니다.
    """
    
    def __init__(self, uploads_base_path: str = None):
        """
        Args:
            uploads_base_path: uploads 폴더 경로 (기본값: ../uploads)
        """
        if not HAS_EZDXF:
            raise ImportError("ezdxf 라이브러리가 필요합니다. pip install ezdxf로 설치해주세요.")
        
        # uploads 폴더 경로 설정
        if uploads_base_path is None:
            # 현재 파일 위치에서 상위 폴더의 uploads 찾기
            current_file = Path(__file__)
            project_root = current_file.parent.parent  # src의 상위 폴더
            uploads_base_path = project_root / "uploads"
        
        self.uploads_path = Path(uploads_base_path)
        self.supported_extensions = {'.dwg', '.dxf'}
        
        if not self.uploads_path.exists():
            raise FileNotFoundError(f"uploads 폴더를 찾을 수 없습니다: {self.uploads_path}")
            
        logger.info(f"DWG 프로젝트 처리기 초기화: {self.uploads_path}")
    
    def find_project_folders(self) -> List[Dict[str, Any]]:
        """uploads 폴더에서 프로젝트 폴더 목록 반환"""
        projects = []
        
        try:
            for item in self.uploads_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # 시스템 폴더 제외
                    if item.name in ['processing_metadata', 'projects']:
                        continue
                    
                    # DWG/DXF 파일이 있는지 확인
                    dwg_files = self._find_dwg_files_in_project(item)
                    
                    project_info = {
                        'project_name': item.name,
                        'project_path': str(item),
                        'dwg_file_count': len(dwg_files),
                        'dwg_files': dwg_files
                    }
                    projects.append(project_info)
                    
            logger.info(f"발견된 프로젝트: {len(projects)}개")
            return projects
            
        except Exception as e:
            logger.error(f"프로젝트 폴더 검색 실패: {e}")
            return []
    
    def _find_dwg_files_in_project(self, project_path: Path) -> List[Dict[str, Any]]:
        """프로젝트 폴더 내 모든 DWG/DXF 파일 찾기 (XREF 폴더 제외)"""
        dwg_files = []
        
        try:
            # 재귀적으로 모든 하위 폴더에서 DWG/DXF 파일 찾기
            for ext in ['**/*.dwg', '**/*.dxf', '**/*.DWG', '**/*.DXF']:
                found_files = list(project_path.glob(ext))
                
                for file_path in found_files:
                    # XREF 폴더 제외 - 경로에 XREF가 포함된 경우 건너뛰기
                    if 'XREF' in str(file_path).upper():
                        logger.debug(f"XREF 폴더 파일 제외: {file_path}")
                        continue
                    
                    # 상대 경로 계산 (프로젝트 폴더 기준)
                    relative_path = file_path.relative_to(project_path)
                    
                    file_info = {
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'relative_path': str(relative_path),
                        'parent_folder': file_path.parent.name,
                        'file_size': file_path.stat().st_size if file_path.exists() else 0,
                        'file_extension': file_path.suffix.lower()
                    }
                    dwg_files.append(file_info)
            
            # 파일명으로 정렬
            dwg_files.sort(key=lambda x: x['file_name'])
            
            logger.info(f"프로젝트 '{project_path.name}'에서 {len(dwg_files)}개 CAD 파일 발견 (XREF 제외)")
            return dwg_files
            
        except Exception as e:
            logger.error(f"DWG 파일 검색 실패 ({project_path}): {e}")
            return []
    
    def process_project(self, project_name: str) -> Dict[str, Any]:
        """특정 프로젝트의 모든 DWG 파일 처리"""
        projects = self.find_project_folders()
        target_project = None
        
        for project in projects:
            if project['project_name'] == project_name:
                target_project = project
                break
        
        if not target_project:
            logger.error(f"프로젝트를 찾을 수 없습니다: {project_name}")
            return {}
        
        return self._process_project_files(target_project)
    
    def process_all_projects(self) -> Dict[str, Any]:
        """모든 프로젝트 처리"""
        projects = self.find_project_folders()
        
        if not projects:
            logger.warning("처리할 프로젝트가 없습니다.")
            return {}
        
        all_results = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_projects': len(projects),
            'projects': {}
        }
        
        for project in projects:
            logger.info(f"프로젝트 처리 시작: {project['project_name']}")
            
            try:
                result = self._process_project_files(project)
                all_results['projects'][project['project_name']] = result
                
                logger.info(f"프로젝트 처리 완료: {project['project_name']}")
                
            except Exception as e:
                logger.error(f"프로젝트 처리 실패 ({project['project_name']}): {e}")
                all_results['projects'][project['project_name']] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return all_results
    
    def _process_project_files(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """프로젝트의 DWG 파일들 처리"""
        project_name = project_info['project_name']
        dwg_files = project_info['dwg_files']
        
        if not dwg_files:
            return {
                'status': 'no_files',
                'message': 'DWG/DXF 파일이 없습니다.'
            }
        
        parser = DWGParser()
        processed_files = {}
        processing_summary = {
            'project_name': project_name,
            'total_files': len(dwg_files),
            'processed_files': 0,
            'failed_files': 0,
            'processing_errors': []
        }
        
        for file_info in dwg_files:
            file_path = file_info['file_path']
            file_name = file_info['file_name']
            
            try:
                logger.info(f"파일 처리 시작: {file_name}")
                
                if parser.load_file_with_xref(file_path, str(project_path)):
                    # 메타데이터 추출
                    metadata = parser.extract_all_metadata()
                    
                    # LLM 읽기 쉬운 요약 생성
                    summary = parser.generate_llm_readable_summary()
                    
                    processed_files[file_name] = {
                        'file_info': file_info,
                        'metadata': metadata,
                        'summary': summary,
                        'processing_timestamp': datetime.now().isoformat(),
                        'status': 'success'
                    }
                    
                    # 개별 파일 메타데이터 저장
                    output_dir = Path(project_info['project_path']) / 'processed_metadata'
                    output_dir.mkdir(exist_ok=True)
                    
                    metadata_file = output_dir / f"{Path(file_name).stem}_metadata.json"
                    parser.save_metadata_to_json(str(metadata_file))
                    
                    processing_summary['processed_files'] += 1
                    logger.info(f"파일 처리 완료: {file_name}")
                    
                else:
                    processing_summary['failed_files'] += 1
                    processing_summary['processing_errors'].append(f"파일 로드 실패: {file_name}")
                    logger.error(f"파일 로드 실패: {file_name}")
                    
            except Exception as e:
                processing_summary['failed_files'] += 1
                error_msg = f"파일 처리 오류 ({file_name}): {str(e)}"
                processing_summary['processing_errors'].append(error_msg)
                logger.error(error_msg)
        
        # 프로젝트 통합 결과 저장
        project_result = {
            'project_info': project_info,
            'processing_summary': processing_summary,
            'processed_files': processed_files,
            'status': 'completed'
        }
        
        # 프로젝트 통합 메타데이터 파일 저장
        self._save_project_results(project_name, project_result)
        
        return project_result
    
    def _save_project_results(self, project_name: str, results: Dict[str, Any]):
        """프로젝트 결과를 JSON 파일로 저장"""
        try:
            output_dir = self.uploads_path / project_name / 'processed_metadata'
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"project_analysis_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"프로젝트 결과 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"프로젝트 결과 저장 실패: {e}")
    
    def get_project_summary(self) -> Dict[str, Any]:
        """전체 프로젝트 요약 정보 반환"""
        projects = self.find_project_folders()
        
        summary = {
            'uploads_path': str(self.uploads_path),
            'total_projects': len(projects),
            'projects_overview': [],
            'total_dwg_files': 0
        }
        
        for project in projects:
            project_overview = {
                'project_name': project['project_name'],
                'dwg_file_count': project['dwg_file_count'],
                'main_folders': list(set([
                    file_info['parent_folder'] 
                    for file_info in project['dwg_files']
                ]))
            }
            summary['projects_overview'].append(project_overview)
            summary['total_dwg_files'] += project['dwg_file_count']
        
        return summary

    def convert_dwg_to_dxf(self, dwg_file_path: str, output_dir: str = None) -> Optional[str]:
        """
        DWG 파일을 DXF로 변환
        
        Args:
            dwg_file_path: DWG 파일 경로
            output_dir: 출력 디렉토리 (None이면 원본 파일과 같은 디렉토리)
            
        Returns:
            변환된 DXF 파일 경로 또는 None (실패 시)
        """
        if not HAS_EZDXF:
            logger.error("ezdxf 라이브러리가 없습니다.")
            return None
            
        dwg_path = Path(dwg_file_path)
        if not dwg_path.exists():
            logger.error(f"DWG 파일을 찾을 수 없습니다: {dwg_file_path}")
            return None
            
        # 출력 파일 경로 설정
        if output_dir is None:
            output_dir = dwg_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        dxf_file_path = output_dir / f"{dwg_path.stem}.dxf"
        
        # 먼저 ezdxf 내장 변환 방법 시도
        try:
            logger.info(f"ezdxf 내장 변환 시도: {dwg_file_path}")
            
            # ezdxf의 odafc addon을 사용한 변환
            doc = odafc.readfile(str(dwg_path))
            doc.saveas(str(dxf_file_path))
            
            if dxf_file_path.exists():
                logger.info(f"✅ ezdxf 내장 변환 성공: {dxf_file_path}")
                return str(dxf_file_path)
                
        except Exception as e:
            logger.warning(f"ezdxf 내장 변환 실패: {e}")
        
        # ODA File Converter 외부 실행 방법 시도
        try:
            import os
            import subprocess
            import shutil
            
            # ODA File Converter 설치 확인
            oda_converter = shutil.which('ODAFileConverter')
            if not oda_converter:
                logger.warning("ODA File Converter를 찾을 수 없습니다.")
                return None
            
            logger.info(f"ODA File Converter로 변환 시도: {dwg_file_path}")
            
            # 환경 변수 설정
            env = os.environ.copy()
            env.update({
                'QT_QPA_PLATFORM': 'offscreen',  # GUI 없이 실행
                'DISPLAY': ':99',
                'XDG_RUNTIME_DIR': '/tmp',
                'HOME': os.path.expanduser('~')
            })
            
            # 임시 작업 디렉토리 생성
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # DWG 파일을 임시 디렉토리에 복사
                temp_dwg = temp_path / dwg_path.name
                shutil.copy2(dwg_path, temp_dwg)
                
                # ODA File Converter 실행
                cmd = [
                    'timeout', '60',  # 60초 타임아웃
                    'xvfb-run', '-a', '-s', '-screen 0 1024x768x24 -ac +extension GLX',
                    oda_converter,
                    str(temp_path),  # 입력 디렉토리
                    str(temp_path),  # 출력 디렉토리
                    'ACAD2018',  # 출력 버전
                    'DXF',  # 출력 형식
                    '0',  # 재귀적 검색 안함
                    '1',  # 감사 정보 포함
                    '*'  # 모든 파일
                ]
                
                logger.debug(f"실행 명령어: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd, 
                    env=env, 
                    capture_output=True, 
                    text=True,
                    cwd=temp_path,
                    timeout=90
                )
                
                # 변환된 파일 찾기
                temp_dxf = temp_path / f"{dwg_path.stem}.dxf"
                
                if temp_dxf.exists():
                    # 결과 파일을 목적지로 복사
                    shutil.copy2(temp_dxf, dxf_file_path)
                    logger.info(f"✅ ODA File Converter 변환 성공: {dxf_file_path}")
                    return str(dxf_file_path)
                else:
                    logger.error(f"변환 파일이 생성되지 않았습니다: {temp_dxf}")
                    if result.stdout:
                        logger.debug(f"stdout: {result.stdout}")
                    if result.stderr:
                        logger.debug(f"stderr: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logger.error("ODA File Converter 실행 시간 초과")
        except Exception as e:
            logger.error(f"ODA File Converter 변환 실패: {e}")
        
        # 모든 변환 방법 실패
        logger.error(f"❌ 모든 변환 방법 실패: {dwg_file_path}")
        return None
    
    def convert_project_dwg_files(self, project_name: str, output_subdir: str = "converted_dxf") -> Dict[str, str]:
        """
        프로젝트 내 모든 DWG 파일을 DXF로 변환
        
        Args:
            project_name: 프로젝트명
            output_subdir: 변환된 파일을 저장할 하위 디렉토리명
            
        Returns:
            {원본_dwg_경로: 변환된_dxf_경로} 딕셔너리
        """
        conversion_results = {}
        
        try:
            project_path = self.uploads_path / project_name
            if not project_path.exists():
                logger.error(f"프로젝트 경로를 찾을 수 없습니다: {project_path}")
                return conversion_results
            
            # DWG 파일 찾기 (XREF 폴더 제외)
            dwg_files = []
            for dwg_file in project_path.rglob("*.dwg"):
                # XREF 폴더 제외
                if 'XREF' not in str(dwg_file).upper():
                    dwg_files.append(dwg_file)
            
            if not dwg_files:
                logger.info(f"프로젝트에서 DWG 파일을 찾을 수 없습니다: {project_name}")
                return conversion_results
            
            # 출력 디렉토리 설정
            output_dir = project_path / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"프로젝트 '{project_name}'에서 {len(dwg_files)}개 DWG 파일 변환 시작")
            
            for dwg_file in dwg_files:
                try:
                    # 파일별 출력 디렉토리 (원본 구조 유지)
                    relative_path = dwg_file.relative_to(project_path)
                    file_output_dir = output_dir / relative_path.parent
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 변환 수행
                    dxf_path = self.convert_dwg_to_dxf(str(dwg_file), str(file_output_dir))
                    
                    if dxf_path:
                        conversion_results[str(dwg_file)] = dxf_path
                        logger.info(f"✅ 변환 완료: {relative_path}")
                    else:
                        logger.warning(f"❌ 변환 실패: {relative_path}")
                        
                except Exception as e:
                    logger.error(f"파일 변환 중 오류 ({dwg_file}): {e}")
                    continue
            
            logger.info(f"프로젝트 변환 완료: {len(conversion_results)}/{len(dwg_files)} 파일")
            
            # 변환 결과 저장
            results_file = output_dir / "conversion_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'conversion_date': datetime.now().isoformat(),
                    'project_name': project_name,
                    'total_files': len(dwg_files),
                    'converted_files': len(conversion_results),
                    'conversions': {str(k): str(v) for k, v in conversion_results.items()}
                }, f, ensure_ascii=False, indent=2)
            
            return conversion_results
            
        except Exception as e:
            logger.error(f"프로젝트 DWG 변환 실패: {e}")
            return {}
