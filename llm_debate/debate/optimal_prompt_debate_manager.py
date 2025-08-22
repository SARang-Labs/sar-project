"""
최적 프롬프트 토론 관리자

사용자 요구사항에 맞는 새로운 토론 시스템:
1. 각 에이전트가 최초 프롬프트 생성 → 가설 생성
2. 3번의 체계적 토론 (직접 인용 기반 평가)
3. 토론 결과를 종합하여 최종 최적 프롬프트 생성
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json
from ..agents.base_agent import BaseAgent
from ..agents.structural_agent import StructuralAgent
from ..agents.biological_agent import BiologicalAgent

@dataclass
class InitialPromptWithHypothesis:
    """에이전트의 초기 프롬프트와 생성된 가설"""
    agent_name: str
    expertise: str
    initial_prompt: str
    generated_hypothesis: str
    timestamp: float

@dataclass
class DebateRound:
    """토론 라운드 정보"""
    round_number: int
    focus_agent: str  # 이번 라운드에서 집중적으로 평가받는 에이전트
    focus_prompt: str
    focus_hypothesis: str
    evaluations: List[Dict]  # 다른 에이전트들의 평가
    timestamp: float

@dataclass
class DirectQuoteEvaluation:
    """직접 인용 기반 평가"""
    evaluator_agent: str
    target_agent: str
    quoted_text: str  # 직접 인용된 텍스트
    evaluation_type: str  # "praise" 또는 "criticism"
    reasoning: str  # 평가 근거
    improvement_suggestion: str  # 개선 제안 (비판인 경우)

@dataclass
class OptimalPromptDebateState:
    """최적 프롬프트 토론 상태"""
    activity_cliff: Dict = None
    context_info: Dict = None
    target_name: str = ""
    
    # 1단계: 초기 프롬프트 및 가설 생성
    initial_prompts_with_hypotheses: List[InitialPromptWithHypothesis] = field(default_factory=list)
    
    # 2단계: 3번의 토론 라운드
    debate_rounds: List[DebateRound] = field(default_factory=list)
    
    # 3단계: 최종 최적 프롬프트
    final_optimal_prompt: str = ""
    final_optimal_hypothesis: str = ""
    
    # 메타데이터
    debate_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    errors: List[str] = field(default_factory=list)

class OptimalPromptDebateManager:
    """
    최적 프롬프트 토론 관리자
    
    사용자 요구사항을 정확히 구현:
    1. 각 에이전트 최초 프롬프트 + 가설 생성 및 화면 표시
    2. 3개 에이전트 × 3번 토론 (각자를 주제로)
    3. 직접 인용 기반 투명한 평가
    4. 토론 결과 종합하여 최적 프롬프트 1개 생성
    5. 최종 프롬프트 + 가설 전문 제시
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.debate_topic = "자동화된 지능형 SAR 분석 시스템을 위한 최적 근거 중심 가설 생성 방법론 확립"
    
    def setup_agents_unified(self, llm_provider: str, api_key: str):
        """선택된 단일 LLM으로 3개 에이전트 통일 설정"""
        if llm_provider == "OpenAI":
            model_provider = "openai"
            model_name = "gpt-4o"
        elif llm_provider in ["Gemini", "Google Gemini"]:
            model_provider = "gemini"
            model_name = "gemini-2.5-pro"  # 최신 최고 성능 모델
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {llm_provider}")
        
        # 모든 에이전트를 선택된 단일 모델로 통일
        self.agents = {
            "structural": StructuralAgent(
                model_provider=model_provider,
                model_name=model_name,
                api_key=api_key,
                temperature=0.3
            ),
            "biological": BiologicalAgent(
                model_provider=model_provider,
                model_name=model_name,
                api_key=api_key,
                temperature=0.3
            ),
            "sar": BiologicalAgent(  # FutureHouseAgent 대신 BiologicalAgent 사용
                model_provider=model_provider,
                model_name=model_name,
                api_key=api_key,
                temperature=0.3
            )
        }
        
        # SAR 에이전트의 전문성을 SAR로 변경
        self.agents["sar"].expertise = "구조-활성 관계 (SAR) 통합"
    
    def run_optimal_prompt_debate(self, 
                                 activity_cliff: Dict, 
                                 context_info: Dict = None,
                                 target_name: str = "") -> OptimalPromptDebateState:
        """전체 최적 프롬프트 토론 실행"""
        state = OptimalPromptDebateState(
            activity_cliff=activity_cliff,
            context_info=context_info,
            target_name=target_name,
            debate_id=f"optimal_debate_{int(time.time())}",
            start_time=time.time()
        )
        
        try:
            # 1단계: 각 에이전트 초기 프롬프트 + 가설 생성
            state = self._generate_initial_prompts_and_hypotheses(state)
            
            # 2단계: 3번의 토론 라운드 실행
            state = self._execute_three_debate_rounds(state)
            
            # 3단계: 최종 최적 프롬프트 생성
            state = self._generate_final_optimal_prompt(state)
            
        except Exception as e:
            state.errors.append(f"토론 실행 오류: {str(e)}")
        
        state.end_time = time.time()
        return state
    
    def _generate_initial_prompts_and_hypotheses(self, state: OptimalPromptDebateState) -> OptimalPromptDebateState:
        """1단계: 각 에이전트가 초기 프롬프트 생성 → 가설 생성"""
        print("🎯 1단계: 각 에이전트 초기 프롬프트 및 가설 생성")
        
        for agent_name, agent in self.agents.items():
            try:
                print(f"  📝 {agent.expertise} 전문가 - 초기 프롬프트 생성 중...")
                
                # 1-1. 초기 프롬프트 생성
                initial_prompt = self._generate_initial_prompt_for_agent(agent, state.activity_cliff, state.target_name)
                
                # 1-2. 생성된 프롬프트로 가설 생성
                hypothesis = self._generate_hypothesis_with_prompt(initial_prompt, state.activity_cliff, state.context_info, state.target_name)
                
                # 1-3. 결과 저장
                initial_data = InitialPromptWithHypothesis(
                    agent_name=agent_name,
                    expertise=agent.expertise,
                    initial_prompt=initial_prompt,
                    generated_hypothesis=hypothesis,
                    timestamp=time.time()
                )
                state.initial_prompts_with_hypotheses.append(initial_data)
                
                print(f"  ✅ {agent.expertise} 프롬프트 및 가설 생성 완료")
                
            except Exception as e:
                state.errors.append(f"{agent_name} 초기 프롬프트 생성 실패: {str(e)}")
                print(f"  ❌ {agent.expertise} 생성 실패: {str(e)}")
        
        print(f"🎯 1단계 완료: {len(state.initial_prompts_with_hypotheses)}개 프롬프트+가설 생성")
        return state
    
    def _generate_initial_prompt_for_agent(self, agent: BaseAgent, activity_cliff: Dict, target_name: str = "") -> str:
        """에이전트별 초기 프롬프트 생성"""
        
        target_protein_info = f"**타겟 단백질**: {target_name}" if target_name else "**타겟 단백질**: 정보 없음"
        
        prompt_generation_instruction = f"""
당신은 {agent.expertise} 전문가로서, Activity Cliff 분석을 위한 최적의 프롬프트를 설계해야 합니다.

**토론 주제**: {self.debate_topic}

**Activity Cliff 데이터**:
- 화합물 1: {activity_cliff['mol_1']['ID']} (pKi: {activity_cliff['mol_1']['pKi']})
- 화합물 2: {activity_cliff['mol_2']['ID']} (pKi: {activity_cliff['mol_2']['pKi']})
- 구조 유사도: {activity_cliff['similarity']:.3f}
- 활성도 차이: {activity_cliff['activity_diff']:.2f}
{target_protein_info}

**당신의 전문 분야**: {agent.expertise}

**핵심 목표**: 신약 개발 전문가가 수행하던 수기 SAR 분석을 자동화하는 것이 목표입니다. 실제 신약 개발에서 활용 가능한 실용적 가설을 생성해야 합니다.

**반드시 포함해야 할 핵심 요소**:
1. **활성 변화 주요 요인 (Key Activity Cliff)**: 활성도 차이의 핵심 원인을 명확히 식별
2. **주요 구조 변경 트렌드 (General SAR Trend)**: 구조 변경이 활성에 미치는 일반적 패턴
3. **타겟 단백질 정보 활용**: 제공된 타겟 단백질 특성을 가설 생성 근거로 반드시 활용

**가설 형식 요구사항**:
가설은 반드시 다음 형식의 도입부로 시작해야 합니다:
```
- **화합물 1 ({activity_cliff['mol_1']['ID']}):** pKi = {activity_cliff['mol_1']['pKi']}
- **화합물 2 ({activity_cliff['mol_2']['ID']}):** pKi = {activity_cliff['mol_2']['pKi']}
- **구조 유사도:** {activity_cliff['similarity']:.3f} (사실상 동일 구조)
- **활성도 차이:** {activity_cliff['activity_diff']:.2f} pKi 단위 (약 {10**(activity_cliff['activity_diff']):.0f}배 차이)
{target_protein_info}
```

**요구사항**:
1. 당신의 전문 분야 관점에서 Activity Cliff 현상을 분석할 수 있는 최적의 프롬프트를 설계하세요
2. 프롬프트는 반드시 5단계 CoT(Chain of Thought) 구조를 포함해야 합니다
3. 구체적이고 검증 가능한 가설을 생성할 수 있어야 합니다
4. 과학적 근거와 실험적 검증 방법을 포함해야 합니다
5. 타겟 단백질 정보를 가설 생성 근거로 활용하도록 지시해야 합니다
6. "활성 변화 주요 요인"과 "구조 변경 트렌드"를 핵심으로 제시하도록 지시해야 합니다

**프롬프트 형식**:
```
당신은 [전문 분야] 전문가입니다...

다음 5단계로 분석해주세요:
1단계: [구체적 지시]
2단계: [구체적 지시]
3단계: [구체적 지시]
4단계: [구체적 지시]
5단계: [구체적 지시]

[추가 지침들...]
```

**{agent.expertise} 전문가로서 최적의 프롬프트를 설계하세요:**
"""
        
        try:
            response = agent._call_llm("당신은 프롬프트 설계 전문가입니다.", prompt_generation_instruction)
            return response
        except Exception as e:
            return f"프롬프트 생성 실패: {str(e)}"
    
    def _generate_hypothesis_with_prompt(self, prompt: str, activity_cliff: Dict, context_info: Dict, target_name: str = "") -> str:
        """생성된 프롬프트로 가설 생성"""
        try:
            # Gemini를 사용하여 가설 생성
            try:
                import google.generativeai as genai
                # API 키 설정 (biological 에이전트의 API 키 사용)
                gemini_agent = self.agents.get("biological")
                if gemini_agent and gemini_agent.api_key:
                    genai.configure(api_key=gemini_agent.api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
            except ImportError:
                return "Google Gemini API 라이브러리가 설치되지 않았습니다. pip install google-generativeai 를 실행해주세요."
            
            activity_cliff_description = f"""
Activity Cliff 정보:
- 화합물 1: {activity_cliff['mol_1']['ID']} (SMILES: {activity_cliff['mol_1']['SMILES']}, pKi: {activity_cliff['mol_1']['pKi']})  
- 화합물 2: {activity_cliff['mol_2']['ID']} (SMILES: {activity_cliff['mol_2']['SMILES']}, pKi: {activity_cliff['mol_2']['pKi']})
- 구조 유사도: {activity_cliff['similarity']:.3f}
- 활성도 차이: {activity_cliff['activity_diff']:.2f}
"""
            
            if target_name:
                activity_cliff_description += f"- 타겟 단백질: {target_name}\n"
            
            context_text = ""
            if context_info and context_info.get('related_papers'):
                context_text = "\n관련 문헌 정보:\n" + "\n".join([
                    f"- {paper.get('title', '')}: {paper.get('abstract', '')[:200]}..." 
                    for paper in context_info['related_papers'][:3]
                ])
            
            # 가설 생성만을 위한 명확한 지시사항
            hypothesis_generation_instruction = f"""
**중요**: 위의 프롬프트를 사용하여 실제 가설을 생성하세요. 프롬프트 자체를 반복하지 말고, 프롬프트에 따라 구체적인 가설을 작성하세요.

**가설 생성 형식**:
- **화합물 1 ({activity_cliff['mol_1']['ID']}):** pKi = {activity_cliff['mol_1']['pKi']}
- **화합물 2 ({activity_cliff['mol_2']['ID']}):** pKi = {activity_cliff['mol_2']['pKi']}
- **구조 유사도:** {activity_cliff['similarity']:.3f} (사실상 동일 구조)
- **활성도 차이:** {activity_cliff['activity_diff']:.2f} pKi 단위 (약 {10**(activity_cliff['activity_diff']):.0f}배 차이)
{f"- **타겟 단백질**: {target_name}" if target_name else ""}

**반드시 다음 두 핵심 요소를 포함한 구체적인 가설을 작성하세요**:
1. **활성 변화 주요 요인 (Key Activity Cliff)**: [구체적인 분석]
2. **주요 구조 변경 트렌드 (General SAR Trend)**: [구체적인 분석]

**중요**: 가설만 작성하고, 프롬프트 내용은 반복하지 마세요.
"""
            
            full_prompt = f"{prompt}\n\n{activity_cliff_description}\n{context_text}\n\n{hypothesis_generation_instruction}"
            
            response = model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            return f"가설 생성 실패: {str(e)}"
    
    def _execute_three_debate_rounds(self, state: OptimalPromptDebateState) -> OptimalPromptDebateState:
        """2단계: 3번의 토론 라운드 실행"""
        print("🎭 2단계: 3번의 체계적 토론 실행")
        
        # 각 에이전트를 주제로 하는 3번의 토론
        for round_num, focus_agent_name in enumerate(self.agents.keys(), 1):
            print(f"\n  🔄 토론 {round_num}라운드: {self.agents[focus_agent_name].expertise} 전문가 프롬프트 집중 평가")
            
            # 해당 라운드의 주제가 되는 프롬프트와 가설 찾기
            focus_data = next(
                item for item in state.initial_prompts_with_hypotheses 
                if item.agent_name == focus_agent_name
            )
            
            # 다른 2명의 에이전트가 평가
            evaluations = []
            for evaluator_name, evaluator_agent in self.agents.items():
                if evaluator_name != focus_agent_name:
                    print(f"    📊 {evaluator_agent.expertise} → {focus_data.expertise} 평가 중...")
                    
                    evaluation = self._conduct_direct_quote_evaluation(
                        evaluator_agent, focus_data, state
                    )
                    evaluations.append(evaluation)
            
            # 토론 라운드 결과 저장
            debate_round = DebateRound(
                round_number=round_num,
                focus_agent=focus_agent_name,
                focus_prompt=focus_data.initial_prompt,
                focus_hypothesis=focus_data.generated_hypothesis,
                evaluations=evaluations,
                timestamp=time.time()
            )
            state.debate_rounds.append(debate_round)
            
            print(f"  ✅ 토론 {round_num}라운드 완료")
        
        print(f"🎭 2단계 완료: 총 {len(state.debate_rounds)}번의 토론 완료")
        return state
    
    def _conduct_direct_quote_evaluation(self, evaluator_agent: BaseAgent, 
                                        focus_data: InitialPromptWithHypothesis, 
                                        state: OptimalPromptDebateState) -> Dict:
        """직접 인용 기반 투명한 평가 실시"""
        
        evaluation_prompt = f"""
**토론 주제**: {self.debate_topic}

당신은 {evaluator_agent.expertise} 전문가로서, {focus_data.expertise} 전문가가 제안한 프롬프트와 가설을 평가해야 합니다.

**평가 대상 프롬프트**:
```
{focus_data.initial_prompt}
```

**해당 프롬프트로 생성된 가설**:
```  
{focus_data.generated_hypothesis}
```

**평가 지침**:
1. **반드시 직접 인용**하여 평가하세요
2. 칭찬할 부분과 비판할 부분을 명확히 구분하세요
3. 각 평가마다 구체적인 근거를 제시하세요
4. {evaluator_agent.expertise} 전문가 관점에서 평가하세요

**응답 형식**:
```json
{{
    "evaluator_expertise": "{evaluator_agent.expertise}",
    "target_expertise": "{focus_data.expertise}",
    "praise_evaluations": [
        {{
            "quoted_text": "직접 인용된 텍스트",
            "reasoning": "이 부분이 우수한 구체적 이유",
            "score": 8.5
        }}
    ],
    "criticism_evaluations": [
        {{
            "quoted_text": "직접 인용된 텍스트", 
            "reasoning": "이 부분의 문제점과 구체적 이유",
            "improvement_suggestion": "구체적 개선 방안",
            "score": 3.2
        }}
    ],
    "overall_assessment": "전체적인 평가 및 {evaluator_agent.expertise} 관점에서의 의견"
}}
```

**{evaluator_agent.expertise} 전문가로서 투명하고 신뢰할 수 있는 평가를 해주세요:**
"""
        
        try:
            # FutureHouse 에이전트인 경우 JSON 형식 응답을 특별히 요청
            if evaluator_agent.model_provider == "futurehouse":
                system_prompt = f"당신은 {evaluator_agent.expertise} 전문가이며, 투명하고 공정한 평가자입니다. 반드시 정확한 JSON 형식으로만 응답해주세요."
                json_instruction = "\n\n**중요**: 위의 JSON 형식을 정확히 지켜서 응답해주세요. 다른 텍스트는 추가하지 마세요."
                response = evaluator_agent._call_llm(system_prompt, evaluation_prompt + json_instruction)
            else:
                response = evaluator_agent._call_llm(
                    f"당신은 {evaluator_agent.expertise} 전문가이며, 투명하고 공정한 평가자입니다.",
                    evaluation_prompt
                )
            
            # JSON 응답 파싱
            try:
                import re
                import json
                
                # 여러 가지 JSON 패턴 시도
                json_patterns = [
                    r'```json\s*(\{.*?\})\s*```',  # 기본 JSON 코드 블록
                    r'```\s*(\{.*?\})\s*```',      # json 태그 없는 코드 블록
                    r'(\{[^{}]*"evaluator_expertise"[^{}]*\})',  # 기본 JSON 객체 패턴
                ]
                
                evaluation_data = None
                for pattern in json_patterns:
                    json_match = re.search(pattern, response, re.DOTALL)
                    if json_match:
                        try:
                            evaluation_data = json.loads(json_match.group(1))
                            break
                        except json.JSONDecodeError:
                            continue
                
                # JSON 파싱이 모두 실패한 경우 - 특히 FutureHouse 에이전트
                if not evaluation_data:
                    if evaluator_agent.model_provider == "futurehouse":
                        # FutureHouse 응답을 구조화된 형태로 변환
                        evaluation_data = self._parse_futurehouse_evaluation(response, evaluator_agent.expertise, focus_data.expertise)
                    else:
                        # 일반적인 파싱 실패 처리
                        evaluation_data = {
                            "evaluator_expertise": evaluator_agent.expertise,
                            "target_expertise": focus_data.expertise,
                            "raw_evaluation": response,
                            "parse_error": "JSON 형식 파싱 실패"
                        }
                        
            except Exception as e:
                evaluation_data = {
                    "evaluator_expertise": evaluator_agent.expertise,
                    "target_expertise": focus_data.expertise,
                    "raw_evaluation": response,
                    "parse_error": f"JSON 파싱 오류: {str(e)}"
                }
            
            return evaluation_data
            
        except Exception as e:
            return {
                "evaluator_expertise": evaluator_agent.expertise,
                "target_expertise": focus_data.expertise,
                "error": f"평가 실행 실패: {str(e)}"
            }
    
    def _generate_final_optimal_prompt(self, state: OptimalPromptDebateState) -> OptimalPromptDebateState:
        """3단계: 토론 결과를 종합하여 최종 최적 프롬프트 생성"""
        print("🏆 3단계: 토론 결과 종합하여 최종 최적 프롬프트 생성")
        
        try:
            # 모든 토론 결과 수집
            all_praise_parts = []
            all_criticism_parts = []
            
            for debate_round in state.debate_rounds:
                for evaluation in debate_round.evaluations:
                    if 'praise_evaluations' in evaluation:
                        for praise in evaluation['praise_evaluations']:
                            all_praise_parts.append({
                                'source_expertise': debate_round.focus_agent,
                                'evaluator_expertise': evaluation.get('evaluator_expertise'),
                                'quoted_text': praise.get('quoted_text'),
                                'reasoning': praise.get('reasoning'),
                                'score': praise.get('score', 0)
                            })
                    
                    if 'criticism_evaluations' in evaluation:
                        for criticism in evaluation['criticism_evaluations']:
                            all_criticism_parts.append({
                                'source_expertise': debate_round.focus_agent,
                                'evaluator_expertise': evaluation.get('evaluator_expertise'),
                                'quoted_text': criticism.get('quoted_text'),
                                'reasoning': criticism.get('reasoning'),
                                'improvement_suggestion': criticism.get('improvement_suggestion'),
                                'score': criticism.get('score', 0)
                            })
            
            # 심판 에이전트가 최종 프롬프트 생성 
            try:
                import google.generativeai as genai
                # API 키 설정 (biological 에이전트의 API 키 사용)
                gemini_agent = self.agents.get("biological")
                if gemini_agent and gemini_agent.api_key:
                    genai.configure(api_key=gemini_agent.api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
            except ImportError:
                state.errors.append("Google Gemini API 라이브러리가 설치되지 않았습니다.")
                return state
            
            final_prompt_instruction = f"""
**토론 주제**: {self.debate_topic}

당신은 중립적인 심판으로서, 3명의 전문가가 진행한 토론 결과를 종합하여 **단 하나의 최적 프롬프트**를 생성해야 합니다.

**원본 프롬프트들**:
"""
            
            for i, initial_data in enumerate(state.initial_prompts_with_hypotheses, 1):
                final_prompt_instruction += f"""
### {i}. {initial_data.expertise} 전문가 프롬프트:
```
{initial_data.initial_prompt}
```
"""
            
            final_prompt_instruction += f"""

**토론에서 좋은 평가를 받은 부분들**:
"""
            for i, praise in enumerate(all_praise_parts, 1):
                final_prompt_instruction += f"""
{i}. [{praise['source_expertise']}] "{praise['quoted_text']}"
   - 평가자: {praise['evaluator_expertise']}
   - 평가 이유: {praise['reasoning']}
   - 점수: {praise['score']}
   
"""
            
            final_prompt_instruction += f"""

**토론에서 개선이 필요하다고 지적된 부분들**:
"""
            for i, criticism in enumerate(all_criticism_parts, 1):
                final_prompt_instruction += f"""
{i}. [{criticism['source_expertise']}] "{criticism['quoted_text']}"
   - 평가자: {criticism['evaluator_expertise']}
   - 문제점: {criticism['reasoning']}
   - 개선 방안: {criticism['improvement_suggestion']}
   
"""
            
            final_prompt_instruction += """

**최종 과제**:
토론 결과를 종합하여 **세 전문가의 장점을 모두 통합한 단 하나의 최적 프롬프트**를 생성하세요.

**요구사항**:
1. 토론에서 좋은 평가를 받은 부분들을 최대한 활용
2. 지적된 문제점들은 개선 방안을 반영하여 수정
3. 5단계 CoT 구조 필수 유지
4. 구조화학, 생체분자 상호작용, SAR 통합 관점을 모두 포괄
5. 실용적이고 검증 가능한 가설 생성 가능

**최종 최적 프롬프트를 생성해주세요:**
"""
            
            # Gemini로 최종 프롬프트 생성
            
            final_prompt_response = model.generate_content(final_prompt_instruction)
            state.final_optimal_prompt = final_prompt_response.text
            
            # 최종 프롬프트로 가설 생성
            print("  🧪 최종 프롬프트로 최종 가설 생성 중...")
            state.final_optimal_hypothesis = self._generate_hypothesis_with_prompt(
                state.final_optimal_prompt, 
                state.activity_cliff, 
                state.context_info,
                state.target_name
            )
            
            print("🏆 3단계 완료: 최종 최적 프롬프트 및 가설 생성 완료")
            
        except Exception as e:
            state.errors.append(f"최종 프롬프트 생성 실패: {str(e)}")
            print(f"❌ 최종 프롬프트 생성 실패: {str(e)}")
        
        return state
    
    def _parse_futurehouse_evaluation(self, response: str, evaluator_expertise: str, target_expertise: str) -> Dict:
        """FutureHouse 에이전트의 텍스트 응답을 구조화된 평가 형태로 변환"""
        
        # 텍스트에서 칭찬과 비판 부분을 추출하려고 시도
        praise_evaluations = []
        criticism_evaluations = []
        
        lines = response.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 섹션 구분 키워드 감지
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['칭찬', 'praise', '장점', '우수', 'positive']):
                if current_content and current_section:
                    self._add_evaluation_item(current_section, current_content, praise_evaluations, criticism_evaluations)
                current_section = 'praise'
                current_content = []
            elif any(keyword in line_lower for keyword in ['비판', 'criticism', '문제', '개선', 'negative']):
                if current_content and current_section:
                    self._add_evaluation_item(current_section, current_content, praise_evaluations, criticism_evaluations)
                current_section = 'criticism'
                current_content = []
            else:
                current_content.append(line)
        
        # 마지막 섹션 처리
        if current_content and current_section:
            self._add_evaluation_item(current_section, current_content, praise_evaluations, criticism_evaluations)
        
        # 내용이 없는 경우 전체 응답을 일반적인 평가로 처리
        if not praise_evaluations and not criticism_evaluations:
            # 응답 톤 분석하여 칭찬/비판 구분
            if any(keyword in response.lower() for keyword in ['우수', '좋', 'good', 'excellent', '훌륭']):
                praise_evaluations.append({
                    "quoted_text": response[:200] + "..." if len(response) > 200 else response,
                    "reasoning": "전반적으로 긍정적인 평가",
                    "score": 7.0
                })
            else:
                criticism_evaluations.append({
                    "quoted_text": response[:200] + "..." if len(response) > 200 else response,
                    "reasoning": "개선이 필요한 부분들에 대한 지적",
                    "improvement_suggestion": "구체적인 개선 방안 필요",
                    "score": 5.0
                })
        
        return {
            "evaluator_expertise": evaluator_expertise,
            "target_expertise": target_expertise,
            "praise_evaluations": praise_evaluations,
            "criticism_evaluations": criticism_evaluations,
            "overall_assessment": f"{evaluator_expertise} 관점에서의 종합 평가 (FutureHouse 응답 파싱)"
        }
    
    def _add_evaluation_item(self, section_type: str, content_lines: List[str], praise_list: List, criticism_list: List):
        """평가 항목을 해당 리스트에 추가"""
        content = ' '.join(content_lines).strip()
        if not content:
            return
            
        if section_type == 'praise':
            praise_list.append({
                "quoted_text": content[:150] + "..." if len(content) > 150 else content,
                "reasoning": "긍정적 평가 요소",
                "score": 7.5
            })
        elif section_type == 'criticism':
            criticism_list.append({
                "quoted_text": content[:150] + "..." if len(content) > 150 else content,
                "reasoning": "개선이 필요한 부분",
                "improvement_suggestion": "구체적인 개선 방안 검토 필요",
                "score": 4.0
            })