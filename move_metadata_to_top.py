#!/usr/bin/env python3
"""
메타데이터 파일 최상위 정리 스크립트
모든 메타데이터 JSON 파일들을 프로젝트 최상위의 metadata 폴더로 이동
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def move_metadata_to_top():
    """모든 메타데이터 파일들을 최상위 metadata 폴더로 이동"""
    
    project_root = Path("uploads/01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면")
    
    if not project_root.exists():
        print(f"❌ 프로젝트 루트를 찾을 수 없습니다: {project_root}")
        return
    
    print("🔄 메타데이터 파일 최상위 폴더로 이동")
    print("=" * 50)
    
    # 최상위 metadata 폴더 생성
    top_metadata_dir = project_root / "metadata"
    top_metadata_dir.mkdir(exist_ok=True)
    print(f"📁 최상위 metadata 폴더 생성: {top_metadata_dir}")
    
    # 모든 메타데이터 파일 찾기
    metadata_files = list(project_root.rglob("*_metadata.json"))
    
    print(f"📋 발견된 메타데이터 파일: {len(metadata_files)}개")
    
    moved_files = 0
    existing_files = 0
    errors = 0
    
    for metadata_file in metadata_files:
        try:
            # 이미 최상위 metadata 폴더에 있는 파일은 건드리지 않음
            if metadata_file.parent == top_metadata_dir:
                print(f"✅ 이미 최상위에 있음: {metadata_file.name}")
                existing_files += 1
                continue
            
            # 목적지 파일 경로
            dest_file = top_metadata_dir / metadata_file.name
            
            # 파일명 중복 처리
            if dest_file.exists():
                # 원본 파일의 상대 경로를 이용해 고유한 이름 생성
                relative_path = metadata_file.relative_to(project_root)
                # 경로의 폴더명들을 언더스코어로 연결
                path_parts = [part for part in relative_path.parts[:-1] if part != "metadata"]
                prefix = "_".join(path_parts).replace(" ", "_").replace("(", "").replace(")", "")
                
                new_name = f"{prefix}_{metadata_file.name}"
                dest_file = top_metadata_dir / new_name
                
                print(f"📦 이름 변경하여 이동: {metadata_file.name} → {new_name}")
            else:
                print(f"📁 이동: {metadata_file.relative_to(project_root)} → metadata/{metadata_file.name}")
            
            # 파일 이동
            shutil.move(metadata_file, dest_file)
            moved_files += 1
                
        except Exception as e:
            print(f"❌ 오류: {metadata_file.name} - {e}")
            errors += 1
    
    # 빈 metadata 폴더들 정리
    print(f"\n🧹 빈 하위 metadata 폴더 정리 중...")
    removed_dirs = 0
    
    for metadata_dir in project_root.rglob("metadata"):
        if metadata_dir != top_metadata_dir and metadata_dir.is_dir():
            try:
                # 폴더가 비어있으면 삭제
                if not any(metadata_dir.iterdir()):
                    metadata_dir.rmdir()
                    print(f"🗑️  빈 폴더 삭제: {metadata_dir.relative_to(project_root)}")
                    removed_dirs += 1
                else:
                    # 남은 파일들 확인
                    remaining_files = list(metadata_dir.glob("*"))
                    print(f"⚠️  폴더에 파일 남음: {metadata_dir.relative_to(project_root)} ({len(remaining_files)}개)")
            except Exception as e:
                print(f"❌ 폴더 삭제 오류: {metadata_dir} - {e}")
    
    print("\n" + "=" * 50)
    print("🎯 정리 완료")
    print(f"  • 이동된 파일: {moved_files}개")
    print(f"  • 이미 존재: {existing_files}개")
    print(f"  • 오류 발생: {errors}개")
    print(f"  • 삭제된 빈 폴더: {removed_dirs}개")
    
    # 최상위 metadata 폴더 내용 확인
    final_files = list(top_metadata_dir.glob("*.json"))
    print(f"\n📊 최상위 metadata 폴더 최종 상태:")
    print(f"  📁 위치: {top_metadata_dir}")
    print(f"  📋 총 파일 수: {len(final_files)}개")
    
    # 파일 종류별 분류
    file_categories = {}
    for json_file in final_files:
        # A01, A02 등으로 분류
        name_parts = json_file.name.split("-")
        if len(name_parts) > 1:
            category = name_parts[0].split("_")[-1]  # A01, A02 등 추출
            if category not in file_categories:
                file_categories[category] = 0
            file_categories[category] += 1
    
    print(f"\n📈 파일 분류:")
    for category, count in sorted(file_categories.items()):
        print(f"  • {category}: {count}개")

def main():
    """메인 실행 함수"""
    print("🚀 메타데이터 파일 최상위 정리 도구")
    print("=" * 60)
    
    try:
        move_metadata_to_top()
        print("\n✅ 최상위 정리 작업이 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 정리 작업 중 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
