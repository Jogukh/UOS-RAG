#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
건축 도면 분석 결과를 원본 이미지에 시각적으로 표시하는 스크립트
"""

import json
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # macOS에서 GUI 없이 사용
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import io

class ArchitecturalVisualizationGenerator:
    def __init__(self, analysis_file="uploads_analysis_results.json", uploads_dir="uploads"):
        self.analysis_file = analysis_file
        self.uploads_dir = uploads_dir
        self.colors = {
            'wall': '#FF4444',      # 빨간색 - 벽
            'window': '#44FF44',    # 초록색 - 창문
            'stair': '#4444FF',     # 파란색 - 계단
            'column': '#FF44FF',    # 자주색 - 기둥
            'beam': '#FFFF44',      # 노란색 - 보
            'slab': '#44FFFF',      # 청록색 - 슬래브
            'dimension': '#FF8800', # 주황색 - 치수
            'text': '#8844FF',      # 보라색 - 텍스트
            'symbol': '#888888'     # 회색 - 기호
        }
        
    def load_analysis_results(self):
        """분석 결과 JSON 파일 로드"""
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"분석 결과 파일 로드 실패: {e}")
            return None
    
    def parse_raw_response(self, raw_response):
        """raw_response에서 JSON 데이터 추출"""
        try:
            # ```json과 ``` 사이의 내용 추출
            if '```json' in raw_response and '```' in raw_response:
                start = raw_response.find('```json') + 7
                end = raw_response.rfind('```')
                json_str = raw_response[start:end].strip()
                return json.loads(json_str)
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
        return None
    
    def visualize_image_analysis(self, image_data):
        """단일 이미지 분석 결과 시각화"""
        file_name = image_data['file']
        image_path = os.path.join(self.uploads_dir, file_name)
        
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
            
        # 원본 이미지 로드
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return None
            
        # 분석 결과에서 요소 정보 추출
        analysis = image_data.get('analysis', {})
        raw_response = analysis.get('raw_response', '')
        
        # JSON 파싱
        elements_data = self.parse_raw_response(raw_response)
        if not elements_data:
            print("분석 결과에서 요소 정보를 추출할 수 없습니다.")
            return None
            
        # 이미지에 요소들 표시
        annotated_img = self.draw_elements_on_image(img, elements_data)
        
        # 범례 생성
        legend_img = self.create_legend()
        
        # 최종 이미지 결합
        final_img = self.combine_image_and_legend(annotated_img, legend_img)
        
        # 결과 저장
        output_path = self.save_visualization(final_img, file_name)
        
        return output_path
    
    def draw_elements_on_image(self, img, elements_data):
        """이미지에 건축 요소들을 표시"""
        # PIL 이미지를 numpy 배열로 변환
        img_array = np.array(img)
        
        # matplotlib으로 그리기
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(img_array)
        
        # 각 요소 카테고리별로 표시
        for category in ['architectural_elements', 'structural_elements', 'annotation_elements']:
            if category in elements_data:
                elements = elements_data[category]
                for element in elements:
                    self.draw_single_element(ax, element, category)
        
        ax.set_title(f'Building Plan Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # matplotlib figure를 PIL Image로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
    
    def draw_single_element(self, ax, element, category):
        """단일 요소를 이미지에 표시"""
        element_type = element.get('type')
        position = element.get('position', [0, 0])
        description = element.get('description', '')
        content = element.get('content', '')
        
        # 색상 선택
        color = self.colors.get(element_type, '#FF0000')
        
        x, y = position[0], position[1]
        
        # 요소 유형에 따른 다른 표시 방법
        if category == 'architectural_elements':
            # 건축 요소는 큰 원과 라벨
            ax.scatter(x, y, c=color, s=200, alpha=0.7, edgecolors='black', linewidth=2)
            label_text = f'{element_type}'
            ax.annotate(label_text, 
                       (x, y), xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                       fontsize=9, fontweight='bold', color='white')
                       
        elif category == 'structural_elements':
            # 구조 요소는 사각형과 라벨
            ax.scatter(x, y, c=color, s=150, alpha=0.7, marker='s', 
                      edgecolors='black', linewidth=2)
            label_text = f'{element_type}'
            ax.annotate(label_text, 
                       (x, y), xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                       fontsize=8, fontweight='bold', color='white')
                       
        elif category == 'annotation_elements':
            # 주석 요소는 작은 다이아몬드와 내용
            ax.scatter(x, y, c=color, s=100, alpha=0.8, marker='D',
                      edgecolors='black', linewidth=1)
            if content:
                ax.annotate(f'{element_type}: {content}', 
                           (x, y), xytext=(15, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
                           fontsize=7, color='white')
    
    def create_legend(self):
        """범례 이미지 생성"""
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 범례 항목들
        legend_items = [
            ('architectural_elements', 'Architectural Elements'),
            ('wall', 'Wall'),
            ('window', 'Window'),
            ('stair', 'Stair'),
            ('', ''),  # 공백
            ('structural_elements', 'Structural Elements'),
            ('column', 'Column'),
            ('beam', 'Beam'),
            ('slab', 'Slab'),
            ('', ''),  # 공백
            ('annotation_elements', 'Annotation Elements'),
            ('dimension', 'Dimension'),
            ('text', 'Text'),
            ('symbol', 'Symbol')
        ]
        
        y_pos = 0.95
        for item_type, item_name in legend_items:
            if item_type == '':
                y_pos -= 0.05
                continue
                
            if item_type in ['architectural_elements', 'structural_elements', 'annotation_elements']:
                # 카테고리 제목
                ax.text(0.05, y_pos, item_name, fontsize=12, fontweight='bold')
                y_pos -= 0.08
            else:
                # 개별 항목
                color = self.colors.get(item_type, '#000000')
                
                # 마커 그리기
                if item_type in ['column', 'beam', 'slab']:
                    ax.scatter(0.1, y_pos, c=color, s=80, marker='s', alpha=0.7)
                elif item_type in ['dimension', 'text', 'symbol']:
                    ax.scatter(0.1, y_pos, c=color, s=60, marker='D', alpha=0.8)
                else:
                    ax.scatter(0.1, y_pos, c=color, s=100, alpha=0.7)
                
                ax.text(0.2, y_pos, item_name, fontsize=10, va='center')
                y_pos -= 0.07
        
        ax.set_title('Legend', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # matplotlib figure를 PIL Image로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
    
    def combine_image_and_legend(self, main_img, legend_img):
        """메인 이미지와 범례를 결합"""
        # 이미지 크기 조정
        main_width, main_height = main_img.size
        legend_width, legend_height = legend_img.size
        
        # 범례 크기를 메인 이미지 높이에 맞게 조정
        legend_ratio = main_height / legend_height
        new_legend_width = int(legend_width * legend_ratio)
        legend_img = legend_img.resize((new_legend_width, main_height), Image.Resampling.LANCZOS)
        
        # 결합된 이미지 생성
        total_width = main_width + new_legend_width + 20  # 20px 여백
        combined_img = Image.new('RGB', (total_width, main_height), 'white')
        
        # 이미지 붙이기
        combined_img.paste(main_img, (0, 0))
        combined_img.paste(legend_img, (main_width + 20, 0))
        
        return combined_img
    
    def save_visualization(self, img, original_filename):
        """시각화 결과 저장"""
        # 파일명 생성
        name, ext = os.path.splitext(original_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{name}_analyzed_{timestamp}.png"
        output_path = os.path.join(self.uploads_dir, output_filename)
        
        try:
            img.save(output_path, 'PNG', quality=95)
            print(f"시각화 결과 저장됨: {output_path}")
            return output_path
        except Exception as e:
            print(f"이미지 저장 실패: {e}")
            return None
    
    def process_all_images(self):
        """모든 이미지에 대해 시각화 처리"""
        analysis_data = self.load_analysis_results()
        if not analysis_data:
            return []
            
        results = []
        image_files = analysis_data.get('image_files', [])
        
        for image_data in image_files:
            print(f"\n처리 중: {image_data['file']}")
            result = self.visualize_image_analysis(image_data)
            if result:
                results.append(result)
                
        return results

def main():
    """메인 함수"""
    print("🎨 건축 도면 분석 결과 시각화 도구")
    print("=" * 50)
    
    # 시각화 생성기 초기화
    visualizer = ArchitecturalVisualizationGenerator()
    
    # 모든 이미지 처리
    results = visualizer.process_all_images()
    
    print(f"\n✅ 완료! {len(results)}개의 시각화 이미지가 생성되었습니다.")
    for result in results:
        print(f"   📁 {result}")
    
    print("\n📋 범례 설명:")
    print("   🔴 Circle: Architectural Elements (wall, window, stair)")
    print("   🟦 Square: Structural Elements (column, beam, slab)")
    print("   🔹 Diamond: Annotation Elements (dimension, text, symbol)")

if __name__ == "__main__":
    main()
