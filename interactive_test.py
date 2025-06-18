#!/usr/bin/env python3
"""
RAG 챗봇 대화형 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_chatbot import RAGChatbot

def quick_test():
    """빠른 대화형 테스트"""
    print("🏗️ RAG 건축 도면 챗봇 빠른 테스트")
    print("=" * 50)
    
    try:
        # RAG 챗봇 초기화
        chatbot = RAGChatbot()
        
        # 몇 가지 질문 테스트
        test_questions = [
            "주동 입면도에 대해 간단히 설명해주세요",
            "이 프로젝트의 도면 종류는?",
            "건축 도면의 용도는 무엇인가요?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("-" * 40)
            
            response = chatbot.chat(question)
            
            print(f"답변: {response['answer'][:200]}...")  # 처음 200자만 출력
            print(f"검색된 문서 수: {response['search_count']}")
            
            if response['search_results']:
                print("참조 문서:")
                for j, result in enumerate(response['search_results'], 1):
                    metadata = result['metadata']
                    print(f"  {j}. {metadata.get('title', 'N/A')} (유사도: {result['similarity']:.3f})")
        
        print(f"\n✅ 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    quick_test()
