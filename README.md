# LogosAI

LogosAI는 다양한 AI 에이전트를 쉽게 생성하고 관리할 수 있는 Python 라이브러리입니다.

## 설치 방법

```bash
pip install logosai
```

또는 소스에서 직접 설치:

```bash
git clone https://github.com/logosai/logosai.git
cd logosai
pip install -e .
```

## 주요 기능

- 다양한 유형의 AI 에이전트 생성 및 관리
- 에이전트 간 통신 및 작업 분배
- 설정 기반의 유연한 에이전트 구성
- LLM 통합 지원
- 실시간 인터넷 검색 및 데이터 분석
- 작업 분류 및 에이전트 추천

## 기본 사용법

```python
from logosai import (
    LogosAIAgent,
    AgentType,
    AgentConfig,
    get_agent_types
)

# 에이전트 설정 파일 경로 지정 (상대 경로 사용)
config_path = "examples/configs/agents.json"

# 에이전트 생성
config = AgentConfig(
    name="Task Classifier",
    agent_type=AgentType.TASK_CLASSIFIER,
    description="작업 분류 에이전트"
)
agent = LogosAIAgent(config)

# 에이전트 타입 정보 로드
agent_types = get_agent_types(config_path)
```

## 설정 파일 구조

에이전트 설정은 `agents.json` 파일에서 관리됩니다. 현재 지원되는 에이전트 유형:

1. **LLM 검색 에이전트** (`llm_search_agent`)
   - LLM의 내장 지식을 활용한 직접 검색
   - 지식 기반 질의응답
   - 사실 확인 및 개념 설명

2. **데이터 분석 에이전트** (`analysis_agent`)
   - 주제 분석
   - 감정 분석
   - 키워드 추출

3. **작업 분류 에이전트** (`task_classifier_agent`)
   - 사용자 쿼리 분석
   - 적절한 에이전트 추천
   - 작업 유형 분류

4. **인터넷 검색 에이전트** (`internet_agent`)
   - 실시간 웹 검색
   - 웹 스크래핑
   - 최신 정보 수집

5. **계산기 에이전트** (`calculator_agent`)
   - 수학적 계산
   - 단위 변환
   - 공식 계산

설정 파일 예시:
```json
{
    "agents": [
        {
            "agent_id": "task_classifier_agent",
            "name": "작업 분류 에이전트",
            "description": "사용자 쿼리를 분석하여 적절한 에이전트 유형을 추천하는 작업 분류 에이전트입니다.",
            "capabilities": [
                "task_classification",
                "agent_recommendation",
                "query_analysis"
            ],
            "examples": [
                "최신 AI 기술 동향을 찾아줘",
                "이 텍스트의 감정을 분석해줘",
                "파이썬 비동기 프로그래밍에 대해 설명해줘"
            ],
            "parameters": {
                "model": "gpt-4-turbo",
                "temperature": 0.3,
                "confidence_threshold": 0.7
            },
            "metadata": {
                "class_name": "TaskClassifierAgent",
                "module_name": "task_classifier_agent",
                "agent_type": "TASK_CLASSIFIER",
                "version": "1.0.0",
                "requires": ["langchain_openai", "langchain.prompts", "pydantic"]
            }
        }
    ]
}
```

## 주요 컴포넌트

### AgentType
현재 지원되는 에이전트 유형:
- TASK_CLASSIFIER: 작업 분류 에이전트
- LLM_SEARCH: LLM 기반 검색 에이전트
- INTERNET_SEARCH: 실시간 인터넷 검색 에이전트
- ANALYSIS: 데이터 분석 에이전트
- CALCULATOR: 수학 계산 에이전트

### AgentConfig
에이전트 설정을 관리하는 클래스:
- name: 에이전트 이름
- agent_type: 에이전트 유형
- description: 에이전트 설명
- config: 추가 설정 (선택사항)

### LogosAIAgent
기본 에이전트 클래스로 다음 기능 제공:
- 초기화 및 설정
- 쿼리 처리
- 결과 포맷팅
- 에러 처리

### get_agent_types()
에이전트 타입 정보를 로드하는 함수:
```python
def get_agent_types(config_path: str = None) -> Dict[str, Dict[str, Any]]:
    """agents.json에서 에이전트 정보를 로드
    
    Args:
        config_path: agents.json 파일의 경로 (기본값: examples/configs/agents.json)
    """
```

## 디렉토리 구조

```
logosai/
├── __init__.py
├── agent_types.py      # 에이전트 타입 정의
├── config.py           # 설정 관리
├── agent.py           # 기본 에이전트 클래스
└── examples/
    ├── configs/
    │   └── agents.json  # 에이전트 설정
    └── task_classifier_manager.py  # 작업 분류 에이전트 예제
```

## 버전 정보

현재 버전: 0.1.2

## 라이선스

MIT License

## 기여하기

프로젝트에 기여하고 싶으신가요? [기여 가이드라인](CONTRIBUTING.md)을 확인해보세요.

## 문의

문제가 있거나 질문이 있으신가요? [이슈 트래커](https://github.com/logosai/logosai/issues)에 등록해주세요. 

## Agent Market과 LLM 통합

LogosAI는 다양한 AI 에이전트를 쉽게 등록, 검색, 활성화할 수 있는 Agent Market을 제공합니다. 이를 통해 LLM은 ACP(Agent Collaboration Protocol)를 사용하여 필요한 에이전트와 원활하게 협업할 수 있습니다.

### Agent Market 아키텍처

Agent Market은 다음과 같은 구성 요소로 이루어져 있습니다:

1. **에이전트 레지스트리**: 사용 가능한 모든 에이전트의 중앙 저장소
2. **ACP 게이트웨이**: LLM과 에이전트 간의 통신을 중계하는 JSON-RPC 기반 게이트웨이
3. **에이전트 프로비저닝 서비스**: 필요에 따라 에이전트 인스턴스를 동적으로 생성하고 관리
4. **인증 및 권한 관리**: 에이전트 접근 제어 및 사용량 모니터링

```
┌─────────────┐    JSON-RPC     ┌─────────────┐     ┌─────────────┐
│             │   (ACP 프로토콜)  │             │     │  Agent #1   │
│    LLM      │<───────────────>│  ACP 게이트웨이 │<────│  Agent #2   │
│   시스템      │                 │             │     │  Agent #3   │
└─────────────┘                 └──────┬──────┘     └─────────────┘
                                       │
                                       ▼
                               ┌─────────────────┐
                               │  에이전트 레지스트리 │
                               └─────────────────┘
```

### Agent Market 사용 프로세스

1. **에이전트 등록**: 개발자가 다양한 기능을 가진 에이전트를 Market에 등록
2. **에이전트 검색**: LLM이 자연어 명령을 기반으로 필요한 에이전트를 검색
3. **에이전트 활성화**: 선택한 에이전트를 필요에 따라 온디맨드로 활성화
4. **ACP 통신**: 표준화된 JSON-RPC 프로토콜을 통해 LLM이 에이전트와 통신
5. **결과 통합**: 에이전트의 응답을 LLM의 컨텍스트에 통합

### LLM과 Agent Market 통합 예제

```python
import asyncio
from logosai.market import AgentMarket
from logosai.acp import ACPClient

async def main():
    # 에이전트 마켓에 연결
    market = AgentMarket(endpoint="https://market.logosai.com")
    
    # 사용 가능한 에이전트 목록 조회
    agents = await market.list_agents(category="search")
    print(f"사용 가능한 검색 에이전트: {len(agents)}개")
    
    # 에이전트 인스턴스 요청
    agent_instance = await market.provision_agent(
        agent_id="internet_search_agent",
        config={"max_results": 5}
    )
    
    # ACP 클라이언트로 에이전트와 통신
    client = ACPClient(endpoint=agent_instance.endpoint)
    
    # 에이전트에 쿼리 전송
    response = await client.query("파이썬에서 비동기 프로그래밍하는 방법은?")
    
    # 결과 처리
    if response.get("status") == "success":
        print(f"에이전트 응답: {response['result']['content']}")
        
        # 에이전트 사용 완료 후 해제
        await market.release_agent(agent_instance.id)

if __name__ == "__main__":
    asyncio.run(main())
```

### LLM 도구(Tool) 형태로 에이전트 사용

최신 LLM 시스템(예: OpenAI, Claude)에서는 함수 호출 또는 도구(Tool) 기능을 통해 직접 에이전트를 활용할 수 있습니다:

```python
from openai import OpenAI
from logosai.market import AgentMarketTools

# 에이전트 마켓 도구 초기화
market_tools = AgentMarketTools()

# OpenAI 클라이언트 설정
client = OpenAI()

# 에이전트를 도구로 등록
tools = market_tools.get_openai_tools(["internet_search", "weather", "calculator"])

# LLM 호출
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "당신은 필요에 따라 에이전트 도구를 활용하는 도우미입니다."},
        {"role": "user", "content": "서울의 오늘 날씨와 내일 기상예보를 알려주세요."}
    ],
    tools=tools
)
```

### 향후 발전 방향

Agent Market은 다음과 같은 방향으로 발전할 예정입니다:

1. **멀티모달 에이전트 지원**: 텍스트뿐만 아니라 이미지, 오디오 등 다양한 모달리티 처리
2. **에이전트 오케스트레이션**: 복잡한 작업을 위한 여러 에이전트의 자동 조율
3. **에이전트 학습 및 개선**: 사용 패턴을 분석하여 에이전트 성능 자동 최적화
4. **커스텀 에이전트 빌더**: 개발자가 쉽게 커스텀 에이전트를 제작할 수 있는 도구 제공
5. **분산 에이전트 네트워크**: 탈중앙화된 에이전트 생태계 구축 

## 샘플 프로세스 및 함수 호출 경로

LogosAI 에이전트와 ACP를 사용한 전체 프로세스 흐름은 다음과 같습니다. 각 단계에서 사용되는 샘플 파일과 주요 함수 호출 경로를 함께 설명합니다.

### 1. 에이전트 마켓 설정

**프로세스**: 에이전트 마켓에 연결하여 사용 가능한 에이전트 목록을 조회합니다.

```python
from logosai.market import AgentMarket

# 에이전트 마켓 인스턴스 생성
market = AgentMarket(endpoint="https://market.logosai.com")

# 사용 가능한 에이전트 목록 조회
agents = await market.list_agents(category="search")
```

**주요 함수 호출 경로**:
- `AgentMarket.__init__` → `logosai/market/__init__.py:45-59`
- `AgentMarket._ensure_session` → `logosai/market/__init__.py:61-73`
- `AgentMarket.list_agents` → `logosai/market/__init__.py:75-101`

### 2. 에이전트 프로비저닝

**프로세스**: 필요한 에이전트를 선택하고 인스턴스를 프로비저닝합니다.

```python
# 에이전트 인스턴스 요청
agent_instance = await market.provision_agent(
    agent_id="internet_search_agent",
    config={"max_results": 5}
)
```

**주요 함수 호출 경로**:
- `AgentMarket.provision_agent` → `logosai/market/__init__.py:125-151`
- `AgentInstance.from_dict` → `logosai/market/__init__.py:31-38`

### 3. ACP 서버 시작

**프로세스**: 에이전트를 위한 ACP 서버를 시작합니다.

```python
from logosai.acp import ACPServer
from logosai.config import AgentConfig
from logosai.agent_types import AgentType

# 에이전트 설정
agent_config = AgentConfig(
    name="Search Agent",
    agent_type=AgentType.INTERNET_SEARCH,
    description="인터넷 검색을 수행하는 에이전트"
)

# ACP 서버 생성 및 시작
server = ACPServer(
    agent_type=AgentType.INTERNET_SEARCH,
    agent_config=agent_config,
    host="0.0.0.0",
    port=8080
)

await server.start(background=True)
```

**주요 함수 호출 경로**:
- `ACPServer.__init__` → `logosai/acp/server.py:63-112`
- `ACPServer.initialize` → `logosai/acp/server.py:188-230`
- `ACPServer.start` → `logosai/acp/server.py:232-270`
- `LogosAIAgent.initialize` → `logosai/agent.py:188-209`

### 4. ACP 클라이언트 연결

**프로세스**: ACP 클라이언트를 생성하고 서버에 연결합니다.

```python
from logosai.acp import ACPClient

# ACP 클라이언트 생성
client = ACPClient(endpoint=agent_instance.endpoint)

# 에이전트 정보 조회
agent_info = await client.get_agent_info()
```

**주요 함수 호출 경로**:
- `ACPClient.__init__` → `logosai/acp/client.py:25-65`
- `ACPClient._ensure_async_session` → `logosai/acp/client.py:78-85`
- `ACPClient.get_agent_info` → `logosai/acp/client.py:87-93`

### 5. 에이전트 쿼리 실행

**프로세스**: 에이전트에 쿼리를 전송하고 결과를 받습니다.

```python
# 에이전트에 쿼리 전송
response = await client.query("파이썬으로 비동기 프로그래밍하는 방법은?")

# 결과 처리
if response.get("status") == "success":
    print(f"에이전트 응답: {response['result']['content']}")
```

**주요 함수 호출 경로**:
- `ACPClient.query` → `logosai/acp/client.py:121-134`
- `ACPClient.call_method` → `logosai/acp/client.py:161-200`
- `ACPServer._handle_jsonrpc_request` → `logosai/acp/server.py:272-330`
- `LogosAIAgent.process` → `logosai/agent.py:252-291`

### 6. 게이트웨이를 통한 다중 에이전트 접근

**프로세스**: ACP 게이트웨이를 통해 여러 에이전트에 접근합니다.

```python
from logosai.market.gateway import ACPGateway

# 게이트웨이 생성 및 시작
gateway = ACPGateway(host="0.0.0.0", port=8090)
await gateway.start()

# JSON-RPC 클라이언트로 게이트웨이에 접근
from logosai.acp import ACPClient
gateway_client = ACPClient(endpoint=f"http://localhost:8090/gateway")

# 에이전트 목록 조회
agents_result = await gateway_client.call_method("list_agents", {"category": "search"})

# 특정 에이전트 쿼리
query_result = await gateway_client.call_method("query_agent", {
    "agent_id": "internet_search_agent",
    "query": "최신 인공지능 트렌드"
})
```

**주요 함수 호출 경로**:
- `ACPGateway.__init__` → `logosai/market/gateway.py:28-78`
- `ACPGateway.start` → `logosai/market/gateway.py:302-352`
- `ACPGateway._handle_gateway_request` → `logosai/market/gateway.py:117-190`
- `ACPGateway._handle_list_agents` → `logosai/market/gateway.py:192-217`
- `ACPGateway._handle_query_agent` → `logosai/market/gateway.py:219-261`

### 7. LLM 도구 형태로 에이전트 사용

**프로세스**: LLM의 도구(Tool) 기능을 통해 에이전트를 사용합니다.

```python
from openai import OpenAI
from logosai.market import AgentMarketTools

# 에이전트 마켓 도구 초기화
market_tools = AgentMarketTools()

# OpenAI 클라이언트 설정
client = OpenAI()

# 에이전트를 도구로 등록
tools = market_tools.get_openai_tools(["internet_search", "weather"])

# LLM 호출
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "에이전트 도구를 활용하는 도우미입니다."},
        {"role": "user", "content": "서울의 오늘 날씨는?"}
    ],
    tools=tools
)

# 도구 호출이 필요한 경우
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    
    # 도구 실행
    tool_result = await market_tools.execute_tool(tool_name, tool_args)
```

**주요 함수 호출 경로**:
- `AgentMarketTools.__init__` → `logosai/market/__init__.py:205-218`
- `AgentMarketTools.get_openai_tools` → `logosai/market/__init__.py:220-258`
- `AgentMarketTools.execute_tool` → `logosai/market/__init__.py:260-300`
- `AgentMarket.provision_agent` → `logosai/market/__init__.py:125-151`

### 전체 프로세스 흐름도

```
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│                 │          │                 │          │                 │
│    LLM/사용자   │◄────────►│  ACP 게이트웨이  │◄────────►│  에이전트 마켓   │
│                 │          │                 │          │                 │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         │                            │                            │
         │                            ▼                            │
         │                   ┌─────────────────┐                   │
         └──────────────────►│  ACP 클라이언트  │◄──────────────────┘
                             │                 │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │                 │
                             │   ACP 서버     │
                             │                 │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │                 │
                             │   LogosAI 에이전트 │
                             │                 │
                             └─────────────────┘
```

이 프로세스 흐름을 따라 LogosAI SDK의 예제 파일들이 모두 연결되어 동작하며, 다양한 AI 에이전트를 활용한 애플리케이션을 구축할 수 있습니다. 
