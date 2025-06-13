#!/usr/bin/env python3
"""
AutoCAD PDF 도면의 벡터 데이터 기반 건축 요소 분석기
PDF 내부의 벡터 그래픽에서 벽, 문, 창호, 공간 등을 추출하여 구조화된 데이터로 변환
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import fitz  # PyMuPDF
    import numpy as np
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Warning: Dependencies not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class ArchitecturalElement:
    """건축 요소 데이터 클래스"""
    element_type: str  # wall, door, window, space, dimension, text
    coordinates: List[Tuple[float, float]]
    properties: Dict[str, Any]
    confidence: float
    page_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Wall:
    """벽체 요소"""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    thickness: float
    length: float
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Space:
    """공간 요소"""
    boundary: List[Tuple[float, float]]
    area: float
    centroid: Tuple[float, float]
    space_name: str
    space_type: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArchitecturalVectorAnalyzer:
    """AutoCAD PDF 벡터 기반 건축 도면 분석기"""
    
    def __init__(self):
        """초기화"""
        if not HAS_DEPS:
            raise ImportError("Required dependencies not available. Install: pip install PyMuPDF shapely numpy")
        
        # 건축 요소 인식 패턴
        self.wall_patterns = {
            'parallel_lines': {'min_distance': 100, 'max_distance': 300},  # 평행선으로 표현된 벽
            'thick_lines': {'min_thickness': 5, 'max_thickness': 50},      # 두꺼운 선으로 표현된 벽
        }
        
        self.door_patterns = {
            'arc_with_line': {'arc_angle_range': (45, 135)},  # 호와 선으로 표현된 문
            'rectangle_gap': {'gap_size_range': (600, 1200)}, # 벽 사이의 간격
        }
        
        self.window_patterns = {
            'parallel_short_lines': {'length_range': (600, 2000)},  # 평행한 짧은 선들
            'rectangle_with_cross': {'cross_pattern': True},        # 십자가 있는 사각형
        }
        
        # 텍스트 패턴 (한국어 건축 도면)
        self.space_text_patterns = [
            r'거실|침실|방|화장실|욕실|부엌|주방|현관|베란다|다용도실',
            r'living|room|bed|bath|kitchen|entrance|balcony',
            r'\d+㎡|\d+평|\d+호',  # 면적 표시
        ]
        
        self.dimension_patterns = [
            r'\d+(?:\.\d+)?(?:mm|cm|m)',  # 치수 표시
            r'\d+(?:\.\d+)?',             # 숫자만
        ]
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """PDF 도면 전체 분석"""
        try:
            logger.info(f"Starting architectural analysis of: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            analysis_result = {
                "file_path": pdf_path,
                "total_pages": len(doc),
                "pages": [],
                "summary": {
                    "total_walls": 0,
                    "total_doors": 0,
                    "total_windows": 0,
                    "total_spaces": 0,
                    "total_dimensions": 0,
                }
            }
            
            # 각 페이지 분석
            for page_num in range(len(doc)):
                page_result = self.analyze_page(doc[page_num], page_num)
                analysis_result["pages"].append(page_result)
                
                # 요약 정보 업데이트
                for element_type in ["walls", "doors", "windows", "spaces", "dimensions"]:
                    analysis_result["summary"][f"total_{element_type}"] += len(page_result.get(element_type, []))
            
            doc.close()
            logger.info(f"Analysis completed: {analysis_result['summary']}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_page(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """단일 페이지 분석"""
        logger.info(f"Analyzing page {page_num}")
        
        # 벡터 데이터 추출
        vector_data = self.extract_vector_data(page)
        
        # 텍스트 데이터 추출
        text_data = self.extract_text_data(page)
        
        # 건축 요소 분석
        walls = self.detect_walls(vector_data, page_num)
        doors = self.detect_doors(vector_data, page_num)
        windows = self.detect_windows(vector_data, page_num)
        spaces = self.detect_spaces(vector_data, text_data, page_num)
        dimensions = self.detect_dimensions(text_data, page_num)
        
        return {
            "page_number": page_num,
            "page_size": {
                "width": page.rect.width,
                "height": page.rect.height
            },
            "raw_data": {
                "total_lines": len(vector_data.get('lines', [])),
                "total_curves": len(vector_data.get('curves', [])),
                "total_text_blocks": len(text_data)
            },
            "walls": [wall.to_dict() for wall in walls],
            "doors": doors,
            "windows": windows,
            "spaces": [space.to_dict() for space in spaces],
            "dimensions": dimensions,            "analysis_metadata": {
                "wall_detection_confidence": self.calculate_confidence(walls),
                "space_detection_confidence": self.calculate_confidence(spaces),
            }
        }
    
    def extract_vector_data(self, page: fitz.Page) -> Dict[str, Any]:
        """페이지에서 벡터 데이터 추출"""
        vector_data = {
            'lines': [],
            'curves': [],
            'rectangles': [],
            'paths': []
        }
        
        try:
            # PyMuPDF로 Drawing 객체 추출
            drawings = page.get_drawings()
            logger.info(f"Processing {len(drawings)} drawings")
            
            for drawing in drawings:
                # drawing이 딕셔너리인지 확인
                if not isinstance(drawing, dict):
                    logger.warning(f"Unexpected drawing type: {type(drawing)}")
                    continue
                
                # drawing의 메타데이터 추출
                items = drawing.get('items', [])
                stroke_width = drawing.get('width', 1)
                color = drawing.get('color', (0, 0, 0))
                layer = drawing.get('layer', '')
                
                # 각 item 처리 (실제 구조: tuple 형태)
                for item in items:
                    if not isinstance(item, tuple) or len(item) < 3:
                        continue
                    
                    # 튜플 구조: (type, point1, point2, ...)
                    item_type = item[0]
                    
                    if item_type == 'l':  # 직선 (line)
                        if len(item) >= 3:
                            p1 = item[1]
                            p2 = item[2]
                            
                            # Point 객체에서 좌표 추출
                            start = (float(p1.x), float(p1.y)) if hasattr(p1, 'x') else (0, 0)
                            end = (float(p2.x), float(p2.y)) if hasattr(p2, 'x') else (0, 0)
                            
                            vector_data['lines'].append({
                                'start': start,
                                'end': end,
                                'length': self.calculate_distance(start, end),
                                'angle': self.calculate_angle(start, end),
                                'stroke_width': stroke_width,
                                'color': color,
                                'layer': layer
                            })
                    
                    elif item_type == 'c':  # 곡선/호 (curve)
                        if len(item) >= 4:
                            # 곡선의 제어점들 추출
                            points = []
                            for i in range(1, len(item)):
                                if hasattr(item[i], 'x'):
                                    points.append((float(item[i].x), float(item[i].y)))
                            
                            if points:
                                vector_data['curves'].append({
                                    'points': points,
                                    'type': 'curve',
                                    'stroke_width': stroke_width,
                                    'color': color,
                                    'layer': layer
                                })
                    
                    elif item_type == 're':  # 사각형 (rectangle)
                        if len(item) >= 3:
                            rect = item[1]
                            if hasattr(rect, 'x0'):  # fitz.Rect 객체
                                x1, y1, x2, y2 = rect.x0, rect.y0, rect.x1, rect.y1
                                vector_data['rectangles'].append({
                                    'rect': (x1, y1, x2, y2),
                                    'width': abs(x2 - x1),
                                    'height': abs(y2 - y1),
                                    'area': abs(x2 - x1) * abs(y2 - y1),
                                    'layer': layer
                                })
                    
                    elif item_type == 'qu':  # 4각형 (quad)
                        if len(item) >= 2:
                            quad_points = item[1:]
                            points = []
                            for pt in quad_points:
                                if hasattr(pt, 'x'):
                                    points.append((float(pt.x), float(pt.y)))
                            
                            if len(points) >= 4:
                                # 4각형을 직선들로 분해
                                for i in range(4):
                                    start = points[i]
                                    end = points[(i+1) % 4]
                                    
                                    vector_data['lines'].append({
                                        'start': start,
                                        'end': end,
                                        'length': self.calculate_distance(start, end),
                                        'angle': self.calculate_angle(start, end),
                                        'stroke_width': stroke_width,
                                        'color': color,
                                        'layer': layer,
                                        'source': 'quad'
                                    })
        
        except Exception as e:
            logger.error(f"Vector data extraction failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Extracted {len(vector_data['lines'])} lines, {len(vector_data['curves'])} curves")
        return vector_data
    
    def extract_text_data(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """페이지에서 텍스트 데이터 추출"""
        text_data = []
        
        try:
            # 텍스트 블록 추출
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # 텍스트 블록
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_data.append({
                                "text": span["text"].strip(),
                                "bbox": span["bbox"],
                                "font_size": span["size"],
                                "font_name": span["font"],
                                "position": {
                                    "x": (span["bbox"][0] + span["bbox"][2]) / 2,
                                    "y": (span["bbox"][1] + span["bbox"][3]) / 2
                                }
                            })
        
        except Exception as e:
            logger.warning(f"Text data extraction failed: {e}")
        
        return text_data
    
    def detect_walls(self, vector_data: Dict[str, Any], page_num: int) -> List[Wall]:
        """벽체 요소 검출"""
        walls = []
        lines = vector_data.get('lines', [])
        
        # 1. 평행선 패턴으로 벽 검출
        parallel_walls = self.find_parallel_line_walls(lines)
        walls.extend(parallel_walls)
        
        # 2. 두꺼운 선으로 벽 검출
        thick_walls = self.find_thick_line_walls(lines)
        walls.extend(thick_walls)
        
        # 중복 제거
        walls = self.remove_duplicate_walls(walls)
        
        logger.info(f"Page {page_num}: Detected {len(walls)} walls")
        return walls
    
    def find_parallel_line_walls(self, lines: List[Dict]) -> List[Wall]:
        """평행선 패턴으로 벽 검출"""
        walls = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                if self.are_parallel_lines(line1, line2):
                    distance = self.calculate_parallel_distance(line1, line2)
                    
                    # 벽체 두께 범위 확인
                    if (self.wall_patterns['parallel_lines']['min_distance'] <= distance <= 
                        self.wall_patterns['parallel_lines']['max_distance']):
                        
                        wall = Wall(
                            start_point=line1['start'],
                            end_point=line1['end'],
                            thickness=distance,
                            length=line1['length'],
                            properties={
                                'detection_method': 'parallel_lines',
                                'line1': line1,
                                'line2': line2,
                                'confidence': 0.8
                            }
                        )
                        walls.append(wall)
        
        return walls
    
    def find_thick_line_walls(self, lines: List[Dict]) -> List[Wall]:
        """두꺼운 선으로 벽 검출"""
        walls = []
        
        for line in lines:
            thickness = line.get('stroke_width', 1)
            
            if (self.wall_patterns['thick_lines']['min_thickness'] <= thickness <= 
                self.wall_patterns['thick_lines']['max_thickness']):
                
                wall = Wall(
                    start_point=line['start'],
                    end_point=line['end'],
                    thickness=thickness,
                    length=line['length'],
                    properties={
                        'detection_method': 'thick_line',
                        'stroke_width': thickness,
                        'confidence': 0.7
                    }
                )
                walls.append(wall)
        
        return walls
    
    def detect_doors(self, vector_data: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        """문 요소 검출"""
        doors = []
        
        # 호와 선 패턴으로 문 검출
        curves = vector_data.get('curves', [])
        lines = vector_data.get('lines', [])
        
        for curve in curves:
            # 호의 각도가 문 범위에 있는지 확인
            if self.is_door_arc(curve):
                # 근처에 있는 직선 찾기
                nearby_lines = self.find_nearby_lines(curve, lines, max_distance=50)
                
                if nearby_lines:
                    doors.append({
                        'type': 'door',
                        'arc': curve,
                        'adjacent_lines': nearby_lines,
                        'detection_method': 'arc_with_line',
                        'confidence': 0.75,
                        'page_number': page_num
                    })
        
        logger.info(f"Page {page_num}: Detected {len(doors)} doors")
        return doors
    
    def detect_windows(self, vector_data: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        """창호 요소 검출"""
        windows = []
        lines = vector_data.get('lines', [])
        
        # 평행한 짧은 선들로 창문 검출
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                if (self.are_parallel_lines(line1, line2) and 
                    self.is_window_length(line1['length']) and
                    self.is_window_length(line2['length'])):
                    
                    distance = self.calculate_parallel_distance(line1, line2)
                    
                    if 50 <= distance <= 200:  # 창문 프레임 거리
                        windows.append({
                            'type': 'window',
                            'lines': [line1, line2],
                            'width': max(line1['length'], line2['length']),
                            'height': distance,
                            'detection_method': 'parallel_lines',
                            'confidence': 0.7,
                            'page_number': page_num
                        })
        
        logger.info(f"Page {page_num}: Detected {len(windows)} windows")
        return windows
    
    def detect_spaces(self, vector_data: Dict[str, Any], text_data: List[Dict], page_num: int) -> List[Space]:
        """공간 요소 검출"""
        spaces = []
        
        # 텍스트에서 공간 이름 찾기
        space_texts = []
        for text_item in text_data:
            text = text_item['text']
            for pattern in self.space_text_patterns:
                if re.search(pattern, text):
                    space_texts.append(text_item)
                    break
        
        # 각 공간 텍스트 주변의 경계 찾기
        for space_text in space_texts:
            boundary = self.find_space_boundary(space_text, vector_data)
            
            if boundary:
                area = self.calculate_polygon_area(boundary)
                centroid = self.calculate_centroid(boundary)
                
                space = Space(
                    boundary=boundary,
                    area=area,
                    centroid=centroid,
                    space_name=space_text['text'],
                    space_type=self.classify_space_type(space_text['text']),
                    properties={
                        'text_position': space_text['position'],
                        'font_size': space_text['font_size'],
                        'confidence': 0.8
                    }
                )
                spaces.append(space)
        
        logger.info(f"Page {page_num}: Detected {len(spaces)} spaces")
        return spaces
    
    def detect_dimensions(self, text_data: List[Dict], page_num: int) -> List[Dict[str, Any]]:
        """치수 요소 검출"""
        dimensions = []
        
        for text_item in text_data:
            text = text_item['text']
            
            for pattern in self.dimension_patterns:
                if re.search(pattern, text):
                    dimensions.append({
                        'type': 'dimension',
                        'value': text,
                        'position': text_item['position'],
                        'bbox': text_item['bbox'],
                        'font_size': text_item['font_size'],
                        'confidence': 0.9,
                        'page_number': page_num
                    })
                    break
        
        logger.info(f"Page {page_num}: Detected {len(dimensions)} dimensions")
        return dimensions
    
    # 유틸리티 메서드들
    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이의 거리 계산"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def calculate_angle(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """선의 각도 계산 (라디안)"""
        return np.arctan2(end[1] - start[1], end[0] - start[0])
    
    def are_parallel_lines(self, line1: Dict, line2: Dict, tolerance: float = 5.0) -> bool:
        """두 직선이 평행한지 확인"""
        angle1 = line1['angle']
        angle2 = line2['angle']
        
        # 각도 차이 계산 (0-180도 범위에서)
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        return angle_diff <= tolerance
    
    def calculate_parallel_distance(self, line1: Dict, line2: Dict) -> float:
        """평행선 간의 거리 계산"""
        # 첫 번째 선의 중점에서 두 번째 선까지의 거리
        p1_mid = ((line1['start'][0] + line1['end'][0]) / 2, 
                  (line1['start'][1] + line1['end'][1]) / 2)
        
        line2_geom = LineString([line2['start'], line2['end']])
        point = Point(p1_mid)
        
        return point.distance(line2_geom)
    
    def is_door_arc(self, curve: Dict) -> bool:
        """곡선이 문의 호인지 확인"""
        # 곡선이 호 형태이고 각도가 문 범위에 있는지 확인
        # 실제 구현에서는 곡선의 각도와 반지름을 분석
        return True  # 임시 구현
    
    def find_nearby_lines(self, curve: Dict, lines: List[Dict], max_distance: float = 50) -> List[Dict]:
        """곡선 근처의 직선들 찾기"""
        nearby_lines = []
        
        if 'points' not in curve or not curve['points']:
            return nearby_lines
            
        curve_center = curve['points'][0] if curve['points'] else (0, 0)
        
        for line in lines:
            line_center = ((line['start'][0] + line['end'][0]) / 2,
                          (line['start'][1] + line['end'][1]) / 2)
            
            distance = self.calculate_distance(curve_center, line_center)
            if distance <= max_distance:
                nearby_lines.append(line)
        
        return nearby_lines
    
    def is_window_length(self, length: float) -> bool:
        """선분 길이가 창문 크기 범위에 있는지 확인"""
        min_len, max_len = self.window_patterns['parallel_short_lines']['length_range']
        return min_len <= length <= max_len
    
    def find_space_boundary(self, space_text: Dict, vector_data: Dict) -> List[Tuple[float, float]]:
        """공간 텍스트 주변의 경계 찾기"""
        lines = vector_data.get('lines', [])
        text_pos = space_text['position']
        search_radius = 500  # 텍스트 주변 500 픽셀 범위
        
        # 텍스트 주변의 선들 찾기
        boundary_lines = []
        for line in lines:
            line_center = ((line['start'][0] + line['end'][0]) / 2,
                          (line['start'][1] + line['end'][1]) / 2)
            
            distance = self.calculate_distance(text_pos, line_center)
            if distance <= search_radius:
                boundary_lines.append(line)
        
        # 선들을 연결하여 경계 형성 (간단한 구현)
        if len(boundary_lines) >= 3:
            # 텍스트 위치 기준으로 정렬하여 경계 형성
            points = []
            for line in boundary_lines[:4]:  # 최대 4개 선으로 사각형 경계
                points.extend([line['start'], line['end']])
            
            # 중복 제거 및 정렬
            unique_points = list(set(points))
            if len(unique_points) >= 3:
                return unique_points[:4]  # 사각형 경계
        
        return []
    
    def remove_duplicate_walls(self, walls: List[Wall]) -> List[Wall]:
        """중복된 벽 제거"""
        if not walls:
            return walls
            
        unique_walls = []
        tolerance = 50  # 50 픽셀 내의 벽은 동일한 것으로 간주
        
        for wall in walls:
            is_duplicate = False
            
            for existing_wall in unique_walls:
                # 시작점과 끝점의 거리를 비교
                start_dist = self.calculate_distance(wall.start_point, existing_wall.start_point)
                end_dist = self.calculate_distance(wall.end_point, existing_wall.end_point)
                
                # 역방향도 확인
                start_dist_rev = self.calculate_distance(wall.start_point, existing_wall.end_point)
                end_dist_rev = self.calculate_distance(wall.end_point, existing_wall.start_point)
                
                if ((start_dist < tolerance and end_dist < tolerance) or 
                    (start_dist_rev < tolerance and end_dist_rev < tolerance)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_walls.append(wall)
        
        return unique_walls
    
    def calculate_confidence(self, elements: List) -> float:
        """요소들의 평균 신뢰도 계산"""
        if not elements:
            return 0.0
        
        total_confidence = 0
        count = 0
        
        for element in elements:
            if hasattr(element, 'properties') and 'confidence' in element.properties:
                total_confidence += element.properties['confidence']
                count += 1
            elif isinstance(element, dict) and 'confidence' in element:
                total_confidence += element['confidence']
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """다각형의 넓이 계산 (Shoelace formula)"""
        if len(points) < 3:
            return 0.0
        
        try:
            # Shapely를 사용한 간단한 방법
            polygon = Polygon(points)
            return polygon.area
        except Exception:
            # 수동 계산 (Shoelace formula)
            n = len(points)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            return abs(area) / 2.0
    
    def calculate_centroid(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """다각형의 중심점 계산"""
        if not points:
            return (0.0, 0.0)
        
        try:
            # Shapely를 사용한 방법
            polygon = Polygon(points)
            centroid = polygon.centroid
            return (centroid.x, centroid.y)
        except Exception:
            # 수동 계산 (평균값)
            x_sum = sum(p[0] for p in points)
            y_sum = sum(p[1] for p in points)
            count = len(points)
            return (x_sum / count, y_sum / count)
    
    def classify_space_type(self, space_name: str) -> str:
        """공간 이름으로 공간 타입 분류"""
        space_name_lower = space_name.lower()
        
        # 한국어 공간 분류
        if any(keyword in space_name_lower for keyword in ['거실', 'living']):
            return 'living_room'
        elif any(keyword in space_name_lower for keyword in ['침실', '방', 'bed', 'room']):
            return 'bedroom'
        elif any(keyword in space_name_lower for keyword in ['화장실', '욕실', 'bath', 'toilet']):
            return 'bathroom'
        elif any(keyword in space_name_lower for keyword in ['부엌', '주방', 'kitchen']):
            return 'kitchen'
        elif any(keyword in space_name_lower for keyword in ['현관', 'entrance']):
            return 'entrance'
        elif any(keyword in space_name_lower for keyword in ['베란다', 'balcony']):
            return 'balcony'
        elif any(keyword in space_name_lower for keyword in ['다용도실', 'utility']):
            return 'utility_room'
        else:
            return 'other'


def main():
    """테스트용 메인 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not HAS_DEPS:
        print("❌ Required dependencies not available")
        print("Install: pip install PyMuPDF shapely numpy")
        return
    
    # 테스트 파일
    test_file = r"C:\Users\user\Documents\VLM\uploads\architectural-plan.pdf"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print("🏗️ Starting Architectural Vector Analysis")
    print("=" * 50)
    
    # 분석기 초기화
    analyzer = ArchitecturalVectorAnalyzer()
    
    # PDF 분석 실행
    start_time = time.time()
    result = analyzer.analyze_pdf(test_file)
    end_time = time.time()
    
    print(f"\n⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
    
    if "error" in result:
        print(f"❌ Analysis failed: {result['error']}")
        return
    
    # 결과 출력
    summary = result["summary"]
    print(f"\n📊 Analysis Summary:")
    print(f"   📄 Total pages: {result['total_pages']}")
    print(f"   🧱 Walls: {summary['total_walls']}")
    print(f"   🚪 Doors: {summary['total_doors']}")
    print(f"   🪟 Windows: {summary['total_windows']}")
    print(f"   🏠 Spaces: {summary['total_spaces']}")
    print(f"   📏 Dimensions: {summary['total_dimensions']}")
    
    # 상세 결과 출력
    if result["pages"]:
        first_page = result["pages"][0]
        print(f"\n🔍 Page 0 Details:")
        print(f"   Raw data: {first_page['raw_data']}")
        print(f"   Detected elements: {len(first_page['walls'])} walls, {len(first_page['spaces'])} spaces")
    
    # 결과 저장
    output_file = "architectural_vector_analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    import time
    main()
