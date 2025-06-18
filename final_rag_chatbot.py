#!/usr/bin/env python3
"""
건축 도면 RAG 챗봇 - 최종 완성 버전
DWG 파일에서 추출된 메타데이터를 기반으로 하는 질의응답 시스템
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import colorama
from colorama import Fore, Style, Back

# 컬러 출력 초기화
colorama.init()

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedRAGChatbot:
    """향상된 건축 도면 RAG 챗봇"""
    
    def __init__(self, collection_name: str = "test_architectural_metadata"):
        """
        Args:
            collection_name: 사용할 ChromaDB 컬렉션명
        """
        self.collection_name = collection_name
        self.vector_db = None
        self.llm_client = None
        self.conversation_history = []
        self.stats = {}
        
        print(f"{Fore.CYAN}🏗️ 건축 도면 RAG 챗봇 초기화 중...{Style.RESET_ALL}")
        self._initialize_components()
    
    def _initialize_components(self):
        """RAG 시스템 구성요소 초기화"""
        try:
            # 벡터 DB 초기화
            print(f"{Fore.YELLOW}🔧 벡터 데이터베이스 로딩...{Style.RESET_ALL}")
            from src.metadata_vector_db import MetadataVectorDB
            self.vector_db = MetadataVectorDB(collection_name=self.collection_name)
            
            # LLM 클라이언트 초기화
            print(f"{Fore.YELLOW}🤖 AI 언어 모델 로딩...{Style.RESET_ALL}")
            from src.llm_metadata_extractor import LLMMetadataExtractor
            self.llm_extractor = LLMMetadataExtractor()
            
            # 시스템 통계
            self.stats = self.vector_db.get_collection_stats()
            
            print(f"{Fore.GREEN}✅ RAG 시스템 초기화 완료{Style.RESET_ALL}")
            print(f"  📊 벡터 DB: {self.stats['total_vectors']}개 문서")
            print(f"  🧠 LLM: {getattr(self.llm_extractor, 'model_name', 'Ollama Gemma')}") 
            print(f"  🎯 컬렉션: {self.collection_name}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 시스템 초기화 실패: {e}{Style.RESET_ALL}")
            raise
    
    def search_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """쿼리와 관련된 문서 검색"""
        try:
            results = self.vector_db.search_similar_metadata(query, top_k=top_k)
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
            
            # 문서 내용 포함
            if 'content' in result:
                content = result['content']
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
5. 답변은 한국어로 작성하세요

{context}

위 정보를 바탕으로 다음 질문에 답변해주세요:
질문: {query}

답변:"""
            
            # LangChain ChatOllama를 사용하여 답변 생성
            if self.llm_extractor.llm is None:
                return "죄송합니다. 현재 AI 모델에 연결할 수 없습니다."
            
            response = self.llm_extractor.llm.invoke(rag_prompt)
            answer = response.content.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def chat(self, user_query: str, max_results: int = 3) -> Dict[str, Any]:
        """RAG 기반 챗봇 대화"""
        try:
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
            
            return {
                "query": user_query,
                "answer": answer,
                "search_results": search_results,
                "search_count": len(search_results),
                "timestamp": conversation_entry["timestamp"]
            }
            
        except Exception as e:
            return {
                "query": user_query,
                "answer": f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}",
                "search_results": [],
                "search_count": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        return {
            "vector_db_stats": self.stats,
            "conversation_count": len(self.conversation_history),
            "model_name": getattr(self.llm_extractor, 'model_name', 'Unknown'),
            "collection_name": self.collection_name
        }
    
    def save_conversation(self, filename: str = None):
        """대화 기록 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_conversation_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            return filename
        except Exception as e:
            logger.error(f"대화 기록 저장 실패: {e}")
            return None

def display_welcome():
    """환영 메시지 출력"""
    print(f"{Fore.BLUE}{Style.BRIGHT}")
    print("=" * 70)
    print("🏗️  건축 도면 RAG 챗봇 v2.0")
    print("   DWG 파일 메타데이터 기반 질의응답 시스템")
    print("=" * 70)
    print(f"{Style.RESET_ALL}")
    print(f"{Fore.GREEN}💡 이 챗봇은 건축 도면에 대한 질문에 답변합니다.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📋 사용 가능한 명령어:{Style.RESET_ALL}")
    print("   • 'help' 또는 '도움말' - 사용법 보기")
    print("   • 'stats' 또는 '통계' - 시스템 정보 보기")
    print("   • 'save' 또는 '저장' - 대화 기록 저장")
    print("   • 'clear' 또는 '초기화' - 화면 지우기")
    print("   • 'quit', 'exit', '종료' - 챗봇 종료")
    print()

def display_help():
    """도움말 출력"""
    print(f"{Fore.YELLOW}{Style.BRIGHT}📚 사용법 가이드{Style.RESET_ALL}")
    print("-" * 50)
    print("🔍 질문 예시:")
    print("   • '주동 입면도에 대해 설명해주세요'")
    print("   • '이 프로젝트에는 어떤 도면들이 있나요?'")
    print("   • '평면도와 입면도의 차이점은?'")
    print("   • '지하주차장 관련 도면이 있나요?'")
    print("   • '건축 설계 도면의 특징은?'")
    print()
    print("💡 팁:")
    print("   • 구체적인 도면명이나 건축 용어를 사용하면 더 정확한 답변을 받을 수 있습니다.")
    print("   • 여러 질문을 연속으로 할 수 있습니다.")
    print()

def display_stats(chatbot: EnhancedRAGChatbot):
    """시스템 통계 출력"""
    stats = chatbot.get_stats()
    print(f"{Fore.MAGENTA}{Style.BRIGHT}📊 시스템 통계{Style.RESET_ALL}")
    print("-" * 50)
    print(f"🗃️  벡터 DB 문서 수: {stats['vector_db_stats']['total_vectors']}개")
    print(f"🤖 AI 모델: {stats['model_name']}")
    print(f"💬 대화 수: {stats['conversation_count']}회")
    print(f"🎯 컬렉션: {stats['collection_name']}")
    print()

def display_search_results(results: List[Dict[str, Any]]):
    """검색 결과 표시"""
    if not results:
        print(f"{Fore.YELLOW}⚠️  관련 문서를 찾지 못했습니다.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}📋 참조된 문서 ({len(results)}개):{Style.RESET_ALL}")
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        similarity = result['similarity']
        print(f"  {i}. {Fore.WHITE}{metadata.get('title', 'N/A')}{Style.RESET_ALL} "
              f"({Fore.GREEN}관련도: {similarity:.3f}{Style.RESET_ALL})")

def interactive_chat():
    """향상된 대화형 인터페이스"""
    display_welcome()
    
    try:
        # RAG 챗봇 초기화
        chatbot = EnhancedRAGChatbot()
        print(f"{Fore.GREEN}✅ 챗봇이 준비되었습니다!{Style.RESET_ALL}")
        print()
        
        while True:
            # 사용자 입력
            user_input = input(f"{Fore.BLUE}🙋 질문: {Style.RESET_ALL}").strip()
            
            # 명령어 처리
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print(f"\\n{Fore.GREEN}👋 챗봇을 종료합니다. 감사합니다!{Style.RESET_ALL}")
                break
            
            elif user_input.lower() in ['help', '도움말', 'h']:
                display_help()
                continue
            
            elif user_input.lower() in ['stats', '통계']:
                display_stats(chatbot)
                continue
            
            elif user_input.lower() in ['save', '저장']:
                filename = chatbot.save_conversation()
                if filename:
                    print(f"\\n{Fore.GREEN}💾 대화 기록이 저장되었습니다: {filename}{Style.RESET_ALL}\\n")
                continue
            
            elif user_input.lower() in ['clear', '초기화']:
                os.system('clear' if os.name == 'posix' else 'cls')
                display_welcome()
                continue
            
            # 빈 입력 처리
            if not user_input:
                print(f"{Fore.YELLOW}질문을 입력해주세요.{Style.RESET_ALL}")
                continue
            
            # RAG 응답 생성
            print(f"\\n{Fore.YELLOW}🔍 검색 중...{Style.RESET_ALL}")
            response = chatbot.chat(user_input)
            
            # 결과 출력
            print(f"\\n{Fore.GREEN}{Style.BRIGHT}🤖 답변:{Style.RESET_ALL}")
            print(f"{response['answer']}")
            
            # 검색 결과 요약
            print()
            display_search_results(response['search_results'])
            print("-" * 70)
    
    except KeyboardInterrupt:
        print(f"\\n\\n{Fore.YELLOW}👋 사용자에 의해 종료되었습니다.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\\n{Fore.RED}❌ 챗봇 실행 중 오류 발생: {e}{Style.RESET_ALL}")

def quick_demo():
    """빠른 데모 실행"""
    demo_queries = [
        "주동 입면도에 대해 알려주세요",
        "이 프로젝트에는 어떤 도면들이 있나요?",
        "평면도와 입면도의 차이점은?"
    ]
    
    print(f"{Fore.MAGENTA}{Style.BRIGHT}🎬 RAG 챗봇 데모{Style.RESET_ALL}")
    print("=" * 50)
    
    try:
        chatbot = EnhancedRAGChatbot()
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\\n{Fore.CYAN}[데모 {i}] {query}{Style.RESET_ALL}")
            print("-" * 40)
            
            response = chatbot.chat(query)
            
            # 답변 요약 출력 (처음 150자)
            answer_preview = response['answer'][:150] + "..." if len(response['answer']) > 150 else response['answer']
            print(f"{Fore.GREEN}답변:{Style.RESET_ALL} {answer_preview}")
            
            display_search_results(response['search_results'])
        
        print(f"\\n{Fore.GREEN}✅ 데모 완료{Style.RESET_ALL}")
        
        # 대화형 모드로 전환 제안
        if input(f"\\n{Fore.YELLOW}대화형 모드로 전환하시겠습니까? (y/n): {Style.RESET_ALL}").lower() == 'y':
            print()
            interactive_chat()
        
    except Exception as e:
        print(f"{Fore.RED}❌ 데모 실행 실패: {e}{Style.RESET_ALL}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="건축 도면 RAG 챗봇 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python final_rag_chatbot.py                    # 대화형 모드
  python final_rag_chatbot.py --demo             # 데모 실행
  python final_rag_chatbot.py --mode interactive # 대화형 모드
        """
    )
    
    parser.add_argument("--mode", choices=["interactive", "demo"], default="interactive",
                       help="실행 모드 선택")
    parser.add_argument("--demo", action="store_true", 
                       help="데모 모드로 실행")
    
    args = parser.parse_args()
    
    if args.demo or args.mode == "demo":
        quick_demo()
    else:
        interactive_chat()

if __name__ == "__main__":
    main()
