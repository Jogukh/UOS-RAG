import os
import json
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
import fitz  # PyMuPDF

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv()

def show_metadata(docs):
    """문서의 메타데이터를 출력하는 함수"""
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")

def convert_pdf_to_images(pdf_path, image_folder):
    """PDF를 이미지로 변환하는 함수"""
    # 이미지 폴더가 없으면 생성
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        
    doc = fitz.open(pdf_path)
    # 원본 파일명 그대로 사용
    filename = os.path.basename(pdf_path)
    saved_paths = []

    for page_num in range(len(doc)):
        pix = doc.load_page(page_num).get_pixmap(dpi=300)
        # 확장자만 .png로 변경
        output_path = os.path.join(image_folder, f"{filename.rsplit('.', 1)[0]}.png")
        pix.save(output_path)
        # 절대경로로 저장
        saved_paths.append(os.path.abspath(output_path))

    return saved_paths

def extract_drawing_info(docs):
    """도면 정보를 추출하는 함수"""
    draw_info = {}
    
    # Drawing Title - 뒤에서 앞으로 순회하며 처음 발견되면 바로 저장하고 종료
    for i in range(len(docs) - 2, -1, -1):
        content = docs[i].page_content
        if content and "DRAWING TITLE" in content.upper():
            next_content = docs[i + 1].page_content
            draw_info["Drawing_Title"] = next_content
            break

    # Note - 뒤에서 앞으로 순회하며 처음 발견되면 바로 저장하고 종료
    for i in range(len(docs) - 2, -1, -1):
        content = docs[i].page_content
        if content and "NOTE." in content.upper():
            draw_info["Note"] = docs[i].page_content
            break

    return draw_info

def extract_word_counts(docs):
    """단어 빈도수를 추출하는 함수"""
    # X, Y, 숫자로 시작하지 않는 문장들을 저장할 리스트
    filtered_sentences = []

    # PROJECT TITLE 이전의 문서만 필터링
    project_title_index = -1
    for i, doc in enumerate(docs):
        if "PROJECT TITLE" in doc.page_content:
            project_title_index = i
            break

    filtered_docs = docs[:project_title_index] if project_title_index != -1 else docs

    for doc in filtered_docs:
        if not doc.page_content or doc.page_content.isspace():
            continue
            
        sentence = doc.page_content.strip()
        
        # 특수기호 제거
        sentence_clean = re.sub(r"[.,!?\-–—()\[\]{}\\\"'`~@#$%^&*_+=<>|\\/]", " ", sentence)
        sentence_clean = re.sub(r"\s+", " ", sentence_clean).strip()
        
        # X, Y로 시작하거나 첫 단어가 X, Y이거나 숫자로 시작하는 문장 제외
        if not (sentence_clean.upper().startswith(('X','Y')) or 
                any(word.upper().startswith(('X','Y')) or word[0].isdigit() 
                    for word in sentence_clean.split()[:1])):
            filtered_sentences.append(sentence_clean)

        # 빈 문자열이나 공백만 있는 경우 제외
        if not sentence_clean:
            continue

    # 문장 빈도수 계산
    sentence_counts = Counter(filtered_sentences)
    word_counts = {sentence: count for sentence, count in sentence_counts.most_common()}
    
    return word_counts

def process_single_pdf(pdf_path, image_folder, output_folder):
    """단일 PDF 파일을 처리하는 함수 (Self-Query 호환 형식)"""
    try:
        print(f"처리 중: {pdf_path}")
        
        # PDF 로더 초기화
        loader = UnstructuredPDFLoader(pdf_path, mode="elements", languages=["kor", "eng"])
        docs = loader.load()
        
        # 도면 정보 추출
        draw_info = extract_drawing_info(docs)
        
        # 단어 빈도수 추출
        word_counts = extract_word_counts(docs)
        
        # PDF를 이미지로 변환 (image_folder 사용)
        image_paths = convert_pdf_to_images(pdf_path, image_folder)
        
        # 파일 경로에서 프로젝트 이름 추출
        project_name = "Unknown"
        try:
            path_parts = os.path.normpath(pdf_path).split(os.sep)
            uploads_idx = -1
            for i, part in enumerate(path_parts):
                if part == "uploads":
                    uploads_idx = i
                    break
            
            if uploads_idx >= 0 and uploads_idx + 1 < len(path_parts):
                project_name = path_parts[uploads_idx + 1]
        except Exception as e:
            print(f"⚠️  프로젝트 이름 추출 실패 ({pdf_path}): {e}")
        
        # Self-Query 호환 형식으로 변환
        file_name = os.path.basename(pdf_path)
        combined_dict = {
            "content": f"{draw_info.get('Drawing_Title', file_name)} 도면 정보와 관련 데이터",
            "metadata": {
                "drawing_number": "정보 없음",
                "drawing_title": draw_info.get('Drawing_Title', file_name.replace('.pdf', '')),
                "drawing_type": "도면",
                "drawing_category": "구조도면",
                "project_name": project_name,
                "project_address": "정보 없음",
                "file_name": file_name,
                "file_path": pdf_path,
                "page_number": 1,
                "has_tables": False,
                "has_images": len(image_paths) > 0,
                "land_area": None,
                "building_area": None,
                "total_floor_area": None,
                "building_height": None,
                "floors_above": 0,
                "floors_below": 0,
                "parking_spaces": 0,
                "apartment_units": 0,
                "building_coverage_ratio": None,
                "floor_area_ratio": None,
                "structure_type": "정보 없음",
                "main_use": "정보 없음",
                "approval_date": None,
                "design_firm": "정보 없음",
                "construction_firm": "정보 없음",
                "room_list": [],
                "extracted_at": "2025-06-19T00:00:00",
                "extraction_method": "pdf_image_extractor",
                # 기존 데이터 보존
                "legacy_data": {
                    "draw_info": draw_info,
                    "word_counts": word_counts,
                    "image_paths": image_paths
                }
            }
        }
        
        return combined_dict
        
    except Exception as e:
        print(f"오류 발생 ({pdf_path}): {str(e)}")
        return None

def process_all_pdfs_in_folder(data_folder, image_folder, output_folder):
    """data 폴더의 모든 PDF 파일을 처리하는 함수 (Self-Query 호환)"""
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # data 폴더에서 PDF 파일 찾기
    pdf_files = []
    for file in os.listdir(data_folder):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(data_folder, file))
    
    if not pdf_files:
        print(f"{data_folder} 폴더에 PDF 파일이 없습니다.")
        return
    
    print(f"발견된 PDF 파일 수: {len(pdf_files)}")
    
    # 각 PDF 파일 처리 및 개별 JSON 파일로 저장
    processed_count = 0
    for pdf_path in pdf_files:
        result = process_single_pdf(pdf_path, image_folder, output_folder)
        if result:
            # 개별 JSON 파일로 저장 (Self-Query 호환)
            pdf_basename = os.path.basename(pdf_path).rsplit('.', 1)[0]
            json_filename = f"{pdf_basename}_metadata.json"
            json_path = os.path.join(output_folder, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"✅ {pdf_basename} → {json_filename}")
            processed_count += 1
    
    print(f"처리 완료: {processed_count}개 파일")
    return processed_count

def main():
    """메인 함수"""
    # 현재 작업 디렉토리 확인
    current_dir = os.getcwd()
    print(f"현재 작업 디렉토리: {current_dir}")
    
    # 경로 설정 - 절대 경로로 변경하여 실행 위치에 관계없이 작동하도록 함
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "uploads/부산장안지구_프로젝트도면")
    image_folder = os.path.join(script_dir, "uploads/부산장안지구_프로젝트도면")
    output_folder = os.path.join(script_dir, "uploads/부산장안지구_프로젝트도면")
    
    print(f"스크립트 디렉토리: {script_dir}")
    print(f"데이터 폴더 경로: {data_folder}")
    print(f"이미지 폴더 경로: {image_folder}")
    print(f"출력 폴더 경로: {output_folder}")
    
    # data 폴더가 존재하는지 확인
    if not os.path.exists(data_folder):
        print(f"'{data_folder}' 폴더가 존재하지 않습니다.")
        print("uploads/부산장안지구_프로젝트도면 경로를 찾는 중...")
        
        # uploads/부산장안지구_프로젝트도면 경로 검색
        possible_paths = [
            os.path.join(script_dir, "../uploads/부산장안지구_프로젝트도면"),
            os.path.join(os.path.dirname(script_dir), "uploads/부산장안지구_프로젝트도면"),
            "./uploads/부산장안지구_프로젝트도면",
            "uploads/부산장안지구_프로젝트도면"
        ]
        
        found_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                found_path = abs_path
                print(f"경로를 찾았습니다: {found_path}")
                break
        
        if found_path:
            data_folder = found_path
            image_folder = found_path
            output_folder = found_path
            print(f"모든 작업을 다음 경로에서 수행합니다: {found_path}")
        else:
            print("uploads/부산장안지구_프로젝트도면 폴더를 찾을 수 없습니다.")
            return
    else:
        print(f"데이터 폴더가 존재합니다: {data_folder}")
    
    # 모든 PDF 파일 처리
    results = process_all_pdfs_in_folder(data_folder, image_folder, output_folder)
    
    if results and results > 0:
        print(f"\n처리 완료! 총 {results}개의 PDF 파일이 처리되었습니다.")
        print(f"결과는 '{output_folder}' 폴더에 저장되었습니다.")
        print(f"이미지는 '{image_folder}' 폴더에 저장되었습니다.")
    else:
        print("처리할 PDF 파일이 없거나 오류가 발생했습니다.")

if __name__ == "__main__":
    main()