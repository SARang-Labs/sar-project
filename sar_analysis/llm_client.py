"""
통합 LLM 클라이언트 모듈

이 모듈은 OpenAI와 Google Gemini API를 통합하여 사용할 수 있는 UnifiedLLMClient 클래스를 제공합니다.
SAR 분석 시스템에서 다양한 LLM 공급자를 통해 전문가 시스템의 응답을 생성하는데 사용됩니다.

주요 기능:
- OpenAI GPT-4o 모델 지원
- Google Gemini-2.5-pro 모델 지원
- 공급자 간 일관된 인터페이스 제공
- 온도 조절을 통한 응답 창의성 제어
"""

# === 표준 라이브러리 및 외부 패키지 ===
from openai import OpenAI


class UnifiedLLMClient:
    """
    통합 LLM 클라이언트

    OpenAI와 Google Gemini API를 통합하여 사용할 수 있는 클라이언트 클래스입니다.
    다양한 LLM 공급자에 대해 일관된 인터페이스를 제공하여
    SAR 분석 시스템의 전문가 에이전트들이 활용할 수 있습니다.

    Attributes:
        llm_provider (str): 사용할 LLM 공급자 ("OpenAI" 또는 "Gemini")
        client: LLM API 클라이언트 인스턴스
        model (str): 사용할 모델명
    """

    def __init__(self, api_key: str, llm_provider: str = "OpenAI"):
        """
        UnifiedLLMClient 초기화

        Args:
            api_key (str): LLM 공급자의 API 키
            llm_provider (str): 사용할 LLM 공급자 ("OpenAI" 또는 "Gemini")
                기본값은 "OpenAI"

        Raises:
            ValueError: 지원하지 않는 LLM 공급자가 입력된 경우
        """
        self.llm_provider = llm_provider
        if llm_provider == "OpenAI":
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"
        elif llm_provider in ["Gemini", "Google Gemini"]:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-2.5-pro")
            self.model = "gemini-2.5-pro"
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {llm_provider}")

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """
        통합 응답 생성

        시스템 프롬프트와 사용자 프롬프트를 바탕으로 LLM 응답을 생성합니다.
        공급자에 관계없이 일관된 인터페이스를 제공합니다.

        Args:
            system_prompt (str): 시스템 역할과 지침을 정의하는 프롬프트
            user_prompt (str): 사용자의 질문이나 요청 내용
            temperature (float): 응답의 창의성 조절 파라미터 (0.0-1.0)
                낮을수록 일관된 응답, 높을수록 창의적 응답

        Returns:
            str: LLM이 생성한 응답 텍스트

        Raises:
            ValueError: 지원하지 않는 LLM 공급자인 경우
        """
        if self.llm_provider == "OpenAI":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        elif self.llm_provider in ["Gemini", "Google Gemini"]:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.generate_content(full_prompt)
            return response.text
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {self.llm_provider}")