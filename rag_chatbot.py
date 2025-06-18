#!/usr/bin/env python3
"""
RAG 챗봇 시스템
벡터 DB 기반 검색 + LLM 답변 생성을 통한 건축 도면 질의응답 시스템
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGChatbot:
    """RAG 기반 건축 도면 질의응답 챗봇"""
    
    def __init__(self, collection_name: str = "test_architectural_metadata"):
        """
        Args:
            collection_name: 사용할 ChromaDB 컬렉션명
        """
        self.collection_name = collection_name
        self.vector_db = None
        self.llm_client = None
        self.conversation_history = []
        
        self._initialize_components()
    
    def _initialize_components(self):
        """RAG 시스템 구성요소 초기화"""
        try:
            # 벡터 DB 초기화
            logger.info("🔧 벡터 DB 초기화 중...")
            from src.metadata_vector_db import MetadataVectorDB
            self.vector_db = MetadataVectorDB(collection_name=self.collection_name)
            
            # LLM 클라이언트 초기화
            logger.info("🤖 LLM 클라이언트 초기화 중...")
            from src.llm_metadata_extractor import LLMMetadataExtractor
            self.llm_extractor = LLMMetadataExtractor()
            
            # 벡터 DB 상태 확인
            stats = self.vector_db.get_collection_stats()
            logger.info(f"✅ RAG 시스템 초기화 완료")
            logger.info(f"  - 벡터 DB: {stats['total_vectors']}개 문서")
            logger.info(f"  - LLM 모델: {getattr(self.llm_extractor, 'model_name', 'Ollama LLM')}")
            
        except Exception as e:
            logger.error(f"❌ RAG 시스템 초기화 실패: {e}")
            raise
    
    def search_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """쿼리와 관련된 문서 검색"""
        try:
            logger.info(f"🔍 관련 문서 검색: '{query}'")
            results = self.vector_db.search_similar_metadata(query, top_k=top_k)
            
            logger.info(f"📋 검색 결과: {len(results)}개 문서")
            for i, result in enumerate(results):
                logger.info(f"  {i+1}. {result['metadata'].get('title', 'N/A')} (유사도: {result['similarity']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def format_context_for_llm(self, search_results: List[Dict[str, Any]], query: str) -> str:
        """검색 결과를 LLM용 컨텍스트로 포맷팅"""
        if not search_results:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        context_parts.append(f"사용자 질문: {query}")
        context_parts.append("\\n검색된 관련 건축 도면 정보:")
        
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            similarity = result['similarity']
            
            context_parts.append(f"\\n[문서 {i}] (관련도: {similarity:.3f})")
            context_parts.append(f"제목: {metadata.get('title', 'N/A')}")
            context_parts.append(f"도면 유형: {metadata.get('drawing_type', 'N/A')}")
            context_parts.append(f"파일명: {metadata.get('filename', 'N/A')}")
            context_parts.append(f"프로젝트: {metadata.get('project_name', 'N/A')}")
            
            # 문서 내용 (임베딩 텍스트) 포함
            if 'content' in result:
                content = result['content']
                # 너무 긴 내용은 잘라서 포함
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"내용: {content}")
        
        return "\\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """LLM을 사용하여 답변 생성"""
        try:
            # RAG 시스템용 프롬프트 구성
            rag_prompt = f"""당신은 건축 도면 전문가입니다. 제공된 건축 도면 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.

답변 가이드라인:
1. 검색된 도면 정보를 기반으로 답변하세요
2. 구체적인 도면명, 파일명, 도면 유형을 언급하세요
3. 정보가 부족한 경우 솔직하게 말하세요
4. 건축 전문 용어를 사용하되 이해하기 쉽게 설명하세요

{context}

위 정보를 바탕으로 다음 질문에 답변해주세요:
질문: {query}

답변:"""
            
            logger.info("🤖 LLM 답변 생성 중...")
            
            # LangChain ChatOllama를 사용하여 답변 생성
            if self.llm_extractor.llm is None:
                logger.error("LLM이 초기화되지 않았습니다.")
                return "죄송합니다. 현재 AI 모델에 연결할 수 없습니다."
            
            response = self.llm_extractor.llm.invoke(rag_prompt)
            answer = response.content.strip()
            logger.info("✅ 답변 생성 완료")
            
            return answer
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def chat(self, user_query: str, max_results: int = 3) -> Dict[str, Any]:
        """사용자 질의에 대한 RAG 기반 답변"""
        try:
            logger.info(f"💬 사용자 질의: '{user_query}'")
            
            # 1. 관련 문서 검색
            search_results = self.search_relevant_documents(user_query, top_k=max_results)
            
            # 2. 컨텍스트 구성
            context = self.format_context_for_llm(search_results, user_query)
            
            # 3. LLM 답변 생성
            answer = self.generate_answer(user_query, context)
            
            # 4. 대화 기록 저장
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "search_results_count": len(search_results),
                "answer": answer,
                "search_results": search_results
            }
            self.conversation_history.append(conversation_entry)
            
            logger.info("✅ RAG 응답 완료")
            
            return {
                "query": user_query,
                "answer": answer,
                "search_results": search_results,
                "search_count": len(search_results),
                "timestamp": conversation_entry["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"RAG 챗봇 오류: {e}")
            return {
                "query": user_query,
                "answer": f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}",
                "search_results": [],
                "search_count": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """대화 기록 반환"""
        return self.conversation_history
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        logger.info("대화 기록이 초기화되었습니다.")
    
    def save_conversation(self, filename: str = None):
        """대화 기록을 파일로 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_history_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"대화 기록 저장 완료: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"대화 기록 저장 실패: {e}")
            return None

def interactive_chat():
    """대화형 챗봇 인터페이스"""
    print("🏗️ RAG 건축 도면 챗봇에 오신 것을 환영합니다!")
    print("건축 도면에 대해 무엇이든 질문해보세요.")
    print("종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요.")
    print("대화 기록을 저장하려면 'save'를 입력하세요.")
    print("=" * 60)
    
    try:
        # RAG 챗봇 초기화
        chatbot = RAGChatbot()
        
        while True:
            # 사용자 입력
            user_input = input("\\n🙋 질문: ").strip()
            
            # 종료 명령어
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\\n👋 챗봇을 종료합니다. 감사합니다!")
                break
            
            # 저장 명령어
            if user_input.lower() == 'save':
                filename = chatbot.save_conversation()
                if filename:
                    print(f"\\n💾 대화 기록이 저장되었습니다: {filename}")
                continue
            
            # 빈 입력 처리
            if not user_input:
                print("질문을 입력해주세요.")
                continue
            
            # RAG 응답 생성
            print("\\n🔍 검색 중...")
            response = chatbot.chat(user_input)
            
            # 결과 출력
            print(f"\\n🤖 답변:")
            print(response['answer'])
            
            # 검색 결과 요약
            if response['search_count'] > 0:
                print(f"\\n📋 참조된 문서 ({response['search_count']}개):")
                for i, result in enumerate(response['search_results'], 1):
                    metadata = result['metadata']
                    print(f"  {i}. {metadata.get('title', 'N/A')} (유사도: {result['similarity']:.3f})")
            else:
                print("\\n⚠️  관련 문서를 찾지 못했습니다.")
            
            print("-" * 60)
    
    except KeyboardInterrupt:
        print("\\n\\n👋 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"\\n❌ 챗봇 실행 중 오류 발생: {e}")

def batch_test_queries():
    """배치 테스트 쿼리"""
    test_queries = [
        "주동 입면도에 대해 알려주세요",
        "대지 구적도는 무엇인가요?",
        "평면도와 입면도의 차이점은?",
        "이 프로젝트에는 어떤 도면들이 있나요?",
        "건축 설계 도면의 특징은?",
        "지하주차장 관련 도면이 있나요?"
    ]
    
    print("🧪 배치 테스트 시작")
    print("=" * 60)
    
    try:
        chatbot = RAGChatbot()
        
        for i, query in enumerate(test_queries, 1):
            print(f"\\n[테스트 {i}] {query}")
            print("-" * 40)
            
            response = chatbot.chat(query)
            
            print(f"답변: {response['answer']}")
            print(f"검색된 문서 수: {response['search_count']}")
            
            if response['search_results']:
                print("참조 문서:")
                for j, result in enumerate(response['search_results'], 1):
                    metadata = result['metadata']
                    print(f"  {j}. {metadata.get('title', 'N/A')} (유사도: {result['similarity']:.3f})")
        
        # 테스트 결과 저장
        filename = chatbot.save_conversation()
        print(f"\\n💾 테스트 결과 저장: {filename}")
        
    except Exception as e:
        print(f"❌ 배치 테스트 실패: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 건축 도면 챗봇")
    parser.add_argument("--mode", choices=["interactive", "test"], default="interactive",
                       help="실행 모드: interactive (대화형) 또는 test (배치 테스트)")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_chat()
    elif args.mode == "test":
        batch_test_queries()

if __name__ == "__main__":
    main()
