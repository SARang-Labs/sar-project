"""
최적 프롬프트 토론 인터페이스

사용자 요구사항에 맞는 Streamlit UI:
1. 각 에이전트의 최초 프롬프트와 가설을 토글로 표시
2. 3번의 토론 과정을 체계적으로 시각화
3. 직접 인용 기반 투명한 평가 과정 표시
4. 최종 최적 프롬프트와 가설 전문 깔끔하게 제시
"""

import streamlit as st
import json
from typing import Dict, List, Any, Optional
import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_debate.debate.optimal_prompt_debate_manager import OptimalPromptDebateManager, OptimalPromptDebateState

class OptimalPromptDebateInterface:
    """
    최적 프롬프트 토론 Streamlit 인터페이스
    
    사용자 요구사항을 정확히 구현한 UI 제공
    """
    
    def __init__(self):
        self.debate_manager = OptimalPromptDebateManager()
    
    def show_interface(self, activity_cliff: Dict, context_info: Dict = None, target_name: str = ""):
        """최적 프롬프트 토론 인터페이스 표시"""
        
        # API 키 설정
        api_keys = self._get_api_keys()
        if not self._validate_api_keys(api_keys):
            st.error("API 키를 모두 입력해주세요.")
            return
        
        # 에이전트 설정
        self.debate_manager.setup_agents(api_keys)
        
        # 토론 실행 버튼
        if st.button("최적 프롬프트 토론 시작", type="primary"):
            
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("최적 프롬프트 토론을 진행 중입니다..."):
                
                # 토론 실행
                progress_bar.progress(10)
                
                debate_result = self.debate_manager.run_optimal_prompt_debate(
                    activity_cliff, context_info, target_name
                )
                
                progress_bar.progress(100)
                
                # 결과 표시
                self._display_debate_results(debate_result)
    
    def _get_api_keys(self) -> Dict[str, str]:
        """API 키 입력 받기"""
        st.sidebar.markdown("## 🔑 API 키 설정")
        
        api_keys = {}
        api_keys["openai"] = st.sidebar.text_input("OpenAI API Key", type="password")
        api_keys["gemini"] = st.sidebar.text_input("Google Gemini API Key", type="password")
        api_keys["futurehouse"] = st.sidebar.text_input("FutureHouse API Key", type="password")
        
        return api_keys
    
    def _validate_api_keys(self, api_keys: Dict[str, str]) -> bool:
        """API 키 유효성 검증"""
        return all(api_keys.values())
    
    def _display_debate_results(self, debate_result: OptimalPromptDebateState):
        """토론 결과 전체 표시"""
        
        # 에러 체크
        if debate_result.errors:
            st.error("토론 중 오류 발생:")
            for error in debate_result.errors:
                st.error(f"- {error}")
            return
        
        st.success("최적 프롬프트 토론이 완료되었습니다.")
        
        # 탭으로 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "1️⃣ 초기 프롬프트 & 가설",
            "2️⃣ 토론 과정", 
            "3️⃣ 최종 최적 프롬프트",
            "📊 토론 요약"
        ])
        
        with tab1:
            self._show_initial_prompts_and_hypotheses(debate_result)
        
        with tab2:
            self._show_debate_rounds(debate_result)
        
        with tab3:
            self._show_final_optimal_prompt(debate_result)
        
        with tab4:
            self._show_debate_summary(debate_result)
    
    def _show_initial_prompts_and_hypotheses(self, debate_result: OptimalPromptDebateState):
        """1단계: 각 에이전트의 초기 프롬프트와 가설 표시"""
        
        st.markdown("## 각 전문가의 초기 프롬프트 및 생성된 가설")
        st.markdown("*각 전문가가 독립적으로 생성한 프롬프트와 그 프롬프트로 생성한 가설입니다.*")
        
        for i, initial_data in enumerate(debate_result.initial_prompts_with_hypotheses, 1):
            
            # 전문가별 색상
            color_map = {
                "구조화학": "🔴",
                "생체분자 상호작용": "🟢", 
                "구조-활성 관계 (SAR) 통합": "🔵"
            }
            icon = color_map.get(initial_data.expertise, "⚪")
            
            st.markdown(f"### {icon} {i}. {initial_data.expertise} 전문가")
            
            # 프롬프트 토글
            with st.expander(f"📄 {initial_data.expertise} 전문가가 생성한 프롬프트", expanded=False):
                st.code(initial_data.initial_prompt, language="text")
            
            # 가설 토글
            with st.expander(f"🧪 해당 프롬프트로 생성된 가설", expanded=False):
                st.markdown(initial_data.generated_hypothesis)
            
            st.markdown("---")
    
    def _show_debate_rounds(self, debate_result: OptimalPromptDebateState):
        """2단계: 3번의 토론 과정 표시"""
        
        st.markdown("## 토론 과정 (3라운드)")
        st.markdown("*각 전문가의 프롬프트와 가설에 대해 다른 전문가들이 직접 인용하며 평가합니다.*")
        
        for debate_round in debate_result.debate_rounds:
            
            # 라운드 헤더
            focus_expertise = next(
                (data.expertise for data in debate_result.initial_prompts_with_hypotheses 
                 if data.agent_name == debate_round.focus_agent), 
                "알 수 없음"
            )
            
            st.markdown(f"### 🔄 토론 {debate_round.round_number}라운드: {focus_expertise} 전문가 집중 평가")
            
            with st.expander(f"토론 {debate_round.round_number}라운드 전체 보기", expanded=False):
                
                # 평가 대상 프롬프트와 가설 다시 표시
                st.markdown("#### 📋 평가 대상")
                st.markdown("**프롬프트:**")
                st.code(debate_round.focus_prompt[:500] + "..." if len(debate_round.focus_prompt) > 500 else debate_round.focus_prompt, language="text")
                
                st.markdown("**생성된 가설:**")
                st.info(debate_round.focus_hypothesis[:300] + "..." if len(debate_round.focus_hypothesis) > 300 else debate_round.focus_hypothesis)
                
                st.markdown("#### 🗣️ 전문가들의 평가")
                
                # 각 평가자의 평가 표시
                for evaluation in debate_round.evaluations:
                    evaluator_expertise = evaluation.get('evaluator_expertise', '알 수 없음')
                    
                    st.markdown(f"##### 👨‍🔬 {evaluator_expertise} 전문가의 평가")
                    
                    # JSON 파싱된 평가
                    if 'praise_evaluations' in evaluation and evaluation['praise_evaluations']:
                        st.markdown("**✅ 칭찬받은 부분들:**")
                        for j, praise in enumerate(evaluation['praise_evaluations'], 1):
                            st.success(f"""
**직접 인용**: "{praise.get('quoted_text', 'N/A')}"

**평가 이유**: {praise.get('reasoning', 'N/A')}

**점수**: {praise.get('score', 'N/A')}/10
""")
                    
                    if 'criticism_evaluations' in evaluation and evaluation['criticism_evaluations']:
                        st.markdown("**⚠️ 개선이 필요한 부분들:**")
                        for j, criticism in enumerate(evaluation['criticism_evaluations'], 1):
                            st.warning(f"""
**직접 인용**: "{criticism.get('quoted_text', 'N/A')}"

**문제점**: {criticism.get('reasoning', 'N/A')}

**개선 제안**: {criticism.get('improvement_suggestion', 'N/A')}

**점수**: {criticism.get('score', 'N/A')}/10
""")
                    
                    # 전체 평가
                    if 'overall_assessment' in evaluation:
                        st.markdown("**📝 종합 평가:**")
                        st.markdown(evaluation['overall_assessment'])
                    
                    # 파싱 실패 시 원본 텍스트 표시
                    if 'raw_evaluation' in evaluation:
                        with st.expander("원본 평가 텍스트 (파싱 실패)"):
                            st.text(evaluation['raw_evaluation'])
                    
                    st.markdown("---")
            
            st.markdown("---")
    
    def _show_final_optimal_prompt(self, debate_result: OptimalPromptDebateState):
        """3단계: 최종 최적 프롬프트 및 가설 표시"""
        
        st.markdown("## 🏆 최종 최적 프롬프트 및 가설")
        st.markdown("*토론 결과를 종합하여 생성된 최종 최적 프롬프트와 그것으로 생성한 가설입니다.*")
        
        if not debate_result.final_optimal_prompt:
            st.error("최종 최적 프롬프트가 생성되지 않았습니다.")
            return
        
        # 최종 프롬프트 전문
        st.markdown("### 최종 최적 프롬프트 (전문)")
        with st.container():
            st.markdown("#### 📄 토론을 통해 제안된 최적 프롬프트")
            st.code(debate_result.final_optimal_prompt, language="text")
        
        st.markdown("---")
        
        # 최종 가설 전문  
        st.markdown("### 최종 가설 (전문)")
        if debate_result.final_optimal_hypothesis:
            with st.container():
                st.markdown("#### 📄 최적 프롬프트로 생성된 최종 가설")
                st.markdown(debate_result.final_optimal_hypothesis)
        else:
            st.warning("최종 가설이 생성되지 않았습니다.")
        
        # 다운로드 버튼
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 최종 프롬프트 다운로드",
                data=debate_result.final_optimal_prompt,
                file_name="optimal_prompt.txt",
                mime="text/plain"
            )
        
        with col2:
            if debate_result.final_optimal_hypothesis:
                st.download_button(
                    label="🧪 최종 가설 다운로드", 
                    data=debate_result.final_optimal_hypothesis,
                    file_name="optimal_hypothesis.txt",
                    mime="text/plain"
                )
    
    def _show_debate_summary(self, debate_result: OptimalPromptDebateState):
        """4단계: 토론 요약 통계"""
        
        st.markdown("## 📊 토론 요약")
        
        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("참여 전문가", len(debate_result.initial_prompts_with_hypotheses))
        
        with col2:
            st.metric("토론 라운드", len(debate_result.debate_rounds))
        
        with col3:
            total_duration = debate_result.end_time - debate_result.start_time
            st.metric("총 소요시간", f"{total_duration:.1f}초")
        
        with col4:
            error_count = len(debate_result.errors)
            st.metric("오류 발생", error_count, delta_color="inverse")
        
        # 전문가별 평가 받은 횟수
        st.markdown("### 👥 전문가별 평가 현황")
        
        evaluation_stats = {}
        for debate_round in debate_result.debate_rounds:
            focus_expertise = next(
                (data.expertise for data in debate_result.initial_prompts_with_hypotheses 
                 if data.agent_name == debate_round.focus_agent), 
                "알 수 없음"
            )
            
            praise_count = 0
            criticism_count = 0
            
            for evaluation in debate_round.evaluations:
                praise_count += len(evaluation.get('praise_evaluations', []))
                criticism_count += len(evaluation.get('criticism_evaluations', []))
            
            evaluation_stats[focus_expertise] = {
                'praise': praise_count,
                'criticism': criticism_count
            }
        
        for expertise, stats in evaluation_stats.items():
            st.markdown(f"**{expertise}**:")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"칭찬: {stats['praise']}회")
            with col2:
                st.warning(f"개선점 지적: {stats['criticism']}회")
        
        # 오류 로그
        if debate_result.errors:
            st.markdown("### ⚠️ 발생한 오류들")
            for error in debate_result.errors:
                st.error(error)
    
    def show_sample_interface(self):
        """샘플 데이터로 인터페이스 테스트"""
        st.markdown("# 🧪 최적 프롬프트 토론 시스템 (샘플 테스트)")
        
        # 샘플 Activity Cliff 데이터
        sample_cliff = {
            'mol_1': {
                'ID': 'COMPOUND_001',
                'SMILES': 'CC1=CC=C(C=C1)NC2=NC=NC3=C2C=CN3',
                'pKi': 6.2
            },
            'mol_2': {
                'ID': 'COMPOUND_002',
                'SMILES': 'CC1=CC=C(C=C1)NC2=NC=NC3=C2C=CN3C4CCNCC4',
                'pKi': 8.5
            },
            'similarity': 0.85,
            'activity_diff': 2.3,
            'score': 1.95
        }
        
        st.info("샘플 데이터를 사용하여 인터페이스를 테스트합니다.")
        self.show_interface(sample_cliff, target_name="샘플 타겟")

def main():
    """메인 실행 함수"""
    interface = OptimalPromptDebateInterface()
    interface.show_sample_interface()

if __name__ == "__main__":
    main()