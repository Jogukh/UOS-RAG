#!/usr/bin/env python3
"""
메타데이터 파일 정리 스크립트
기존 _metadata.json 파일들을 metadata 폴더로 이동하고 중복 제거
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def cleanup_metadata_files():
    """기존 메타데이터 파일들을 metadata 폴더로 정리"""
    
    project_root = Path("uploads/01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면/converted_dxf")
    
    if not project_root.exists():
        print(f"❌ 프로젝트 루트를 찾을 수 없습니다: {project_root}")
        return
    
    print("🧹 메타데이터 파일 정리 시작")
    print("=" * 50)
    
    # 1. 기존 _metadata.json 파일들 찾기
    metadata_files = list(project_root.rglob("*_metadata.json"))
    
    print(f"📁 발견된 메타데이터 파일: {len(metadata_files)}개")
    
    moved_files = 0
    kept_files = 0
    removed_duplicates = 0
    
    for metadata_file in metadata_files:
        try:
            # metadata 폴더 안에 있는 파일은 건드리지 않음
            if "metadata" in metadata_file.parts:
                print(f"✅ 유지: {metadata_file.relative_to(project_root)}")
                kept_files += 1
                continue
            
            # 해당 디렉토리에 metadata 폴더 생성
            metadata_dir = metadata_file.parent / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            # 목적지 파일 경로
            dest_file = metadata_dir / metadata_file.name
            
            # 이미 metadata 폴더에 같은 파일이 있는지 확인
            if dest_file.exists():
                # 두 파일의 내용 비교
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f1:
                        content1 = json.load(f1)
                    with open(dest_file, 'r', encoding='utf-8') as f2:
                        content2 = json.load(f2)
                    
                    # 추출 시간 정보 제외하고 비교
                    content1_compare = {k: v for k, v in content1.items() if k != 'extraction_info'}
                    content2_compare = {k: v for k, v in content2.items() if k != 'extraction_info'}
                    
                    if content1_compare == content2_compare:
                        # 내용이 같으면 기존 파일 삭제
                        metadata_file.unlink()
                        print(f"🗑️  중복 제거: {metadata_file.relative_to(project_root)}")
                        removed_duplicates += 1
                    else:
                        # 내용이 다르면 타임스탬프 추가하여 백업
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = f"{metadata_file.stem}_backup_{timestamp}.json"
                        backup_file = metadata_dir / backup_name
                        
                        shutil.move(metadata_file, backup_file)
                        print(f"📦 백업 이동: {metadata_file.relative_to(project_root)} → {backup_file.relative_to(project_root)}")
                        moved_files += 1
                        
                except Exception as e:
                    print(f"⚠️  파일 비교 실패: {metadata_file.name} - {e}")
                    # 오류 시 타임스탬프 추가하여 이동
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{metadata_file.stem}_old_{timestamp}.json"
                    backup_file = metadata_dir / backup_name
                    shutil.move(metadata_file, backup_file)
                    moved_files += 1
            else:
                # metadata 폴더에 파일이 없으면 이동
                shutil.move(metadata_file, dest_file)
                print(f"📁 이동: {metadata_file.relative_to(project_root)} → {dest_file.relative_to(project_root)}")
                moved_files += 1
                
        except Exception as e:
            print(f"❌ 오류: {metadata_file.name} - {e}")
    
    # 2. conversion_results.json은 유지 (변환 결과 로그)
    conversion_file = project_root / "conversion_results.json"
    if conversion_file.exists():
        print(f"📋 변환 로그 유지: {conversion_file.name}")
    
    print("\n" + "=" * 50)
    print("🎯 정리 완료")
    print(f"  • 이동된 파일: {moved_files}개")
    print(f"  • 유지된 파일: {kept_files}개")
    print(f"  • 중복 제거: {removed_duplicates}개")
    
    # 3. 정리된 구조 확인
    print(f"\n📊 정리된 metadata 폴더 구조:")
    metadata_dirs = list(project_root.rglob("metadata"))
    
    for metadata_dir in sorted(metadata_dirs):
        rel_path = metadata_dir.relative_to(project_root)
        json_files = list(metadata_dir.glob("*.json"))
        print(f"  📁 {rel_path}: {len(json_files)}개 파일")
        
        for json_file in sorted(json_files)[:3]:  # 최대 3개만 표시
            print(f"    • {json_file.name}")
        if len(json_files) > 3:
            print(f"    • ... 외 {len(json_files)-3}개")

def main():
    """메인 실행 함수"""
    print("🚀 메타데이터 파일 정리 도구")
    print("=" * 60)
    
    try:
        cleanup_metadata_files()
        print("\n✅ 정리 작업이 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 정리 작업 중 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
