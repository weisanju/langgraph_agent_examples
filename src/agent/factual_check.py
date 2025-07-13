# 事实性检查Agent
import json
from operator import add
from typing import Annotated, TypedDict, List, Literal
from langchain_tavily import TavilySearch

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.types import Send, Command, interrupt

from dotenv import load_dotenv

# 打印 langchain LLm prompt


load_dotenv()


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    search_results: Annotated[List[dict], add_messages]

    final_target: str


llm = ChatTongyi(model="qwen-max-latest")

_search_tool = TavilySearch(max_results=5)


def query_human(query: str,tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    当用户问题模糊、不明确或缺少关键信息时，向用户询问以获取更具体的信息。
    
    :param query: 向用户提出的澄清问题
    :return: 包含用户回复的消息字典
    """
    # 导入
    human_response = interrupt(query)
    inner_human_response = human_response.get(list(human_response.keys())[0])
    answer = inner_human_response['answer']
    print('answer:', answer)

    return Command(
        update={
            "messages": [ToolMessage(answer, tool_call_id=tool_call_id)]
        },
        goto="factual_check")


def create_search_tool_node():
    def do_tool_call(state: State) -> Command[Literal["keypoint_extract", "factual_check"]]:
        # 获取最后一个
        tool_call = state["messages"][-1].tool_calls[0]

        tool_call_id_ = tool_call['id']

        print("tool_call:", tool_call)

        args = tool_call['args']



        if tool_call['name'] == "query_human":
            return query_human(args['query'],tool_call_id_)

        search_results = _search_tool.invoke(tool_call)
        return Command(
            update={
                "search_results": search_results,
                "tool_call_id": tool_call_id_,
            },
            goto=Send('keypoint_extract', {
                "query": args['query'],
                "tool_call_id": tool_call_id_,
                "search_results": search_results.content,
            }))

    return do_tool_call


def create_keypoint_extract():
    user = """
        你的任务是：根据当前问题，对 检索结果 抽取核心知识点，核心知识点要能够 为回答 当前问题 提供支撑
        当前问题：{query}
        检索结果为：{search_results} 
        请仅输出核心关键点，多个用换行符分开。请仅输出 已知的知识信息
        仅抽取与与当前问题相关的核心知识点
    """
    chain = PromptTemplate.from_template(user) | llm

    def keypoint_extract_inner(inner_state):
        return {
            "messages": ToolMessage(chain.invoke(inner_state).content, tool_call_id=inner_state['tool_call_id'])
        }

    return keypoint_extract_inner


def create_factual_check():
    llm_with_tool = llm.bind_tools([_search_tool, query_human])
    system = """
    当前日期为：2025年07月13号

    你是一个问题回答助手，请根据用户的问题，使用知识搜索工具进行检索。你可以多次调用检索工具，直到获取到足够的信息来回答用户的问题。请严格遵循以下要求：

    1. **首轮完整思考**：在开始任何操作之前，请完整思考整个回答步骤与方案：
       
       **问题分析框架**：
       - 核心问题：用户真正想要了解什么？
       - 关键信息点：回答这个问题需要哪些具体信息？
       - 问题清晰度：问题是否明确，是否需要澄清？
       - 信息完整性：是否缺少时间、地点、对象等关键要素？
       
       **回答策略规划**：
       - 所需信息清单：列出所有需要获取的信息点
       - 检索关键词规划：设计有效的搜索关键词
       - 信息获取顺序：确定信息检索的优先级
       - 潜在难点预估：预判可能遇到的信息缺失或模糊点
       
       **执行方案制定**：
       - 是否需要用户澄清：如果问题模糊，优先使用query_human
       - 检索策略：确定搜索工具的使用方式和次数
       - 质量检查点：设定信息完整性的检查标准
       - 最终输出格式：规划回答的结构和呈现方式

    2. **问题澄清阶段**：如果用户问题模糊、不明确或缺少关键信息时，请使用query_human工具向用户询问，获取更具体的信息。以下情况应该使用query_human工具：
       - 时间范围不明确（如"最近"、"以前"、"去年"等）
       - 地点或对象不具体（如"那个地方"、"他们"、"这个公司"等）
       - 问题过于宽泛或需要进一步澄清
       - 存在歧义或多种可能的解释
       - 缺少必要的上下文信息
       使用query_human时，请提出具体、明确的问题来帮助用户澄清他们的需求。

    3. **信息检索阶段**：根据完整思考的结果，使用搜索工具进行知识检索，获取相关信息。

    4. **信息补全阶段**：如果当前检索结果中存在信息缺失、内容不具体或无法直接回答用户问题时，请明确、详细地指出缺失的具体信息，并针对这些缺失点，进一步调用检索工具进行补充检索，直到所有关键信息都被补全。

    5. **质量检查阶段**：检索过程中，每次都要判断现有信息是否足以回答问题，若不足，请继续主动检索缺失的信息内容。绝不能在存在关键信息缺失时停止流程或给出最终回答，必须持续补全所有缺失点。

    6. **内容筛选阶段**：仅回答与用户问题直接相关的信息，避免无关内容。

    7. **最终总结阶段**：只有在所有关键信息都已补全且确认无缺失时，才可以对所有检索到的结果进行归纳总结，给出一个清晰、合理的最终回答。如果仍有信息缺失，请继续检索，不要提前结束。

    **重要提醒**：
    - 每个阶段都要基于前期的完整思考来执行，确保回答的完整性和准确性
    - 首轮思考后，严格按照制定的策略和方案执行，不要偏离原定计划
    - 在执行过程中如果发现新的问题或信息缺失，要及时调整策略并继续补全
    - 始终保持对信息完整性的严格把控，确保最终回答的准确性
    """
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{messages}"),
        ]
    ) | llm_with_tool

    def factual_check(_state: State):
        return {
            "messages": [chain.invoke(_state)],
        }

    return factual_check


def create_factual_check_agent():
    graph = StateGraph(state_schema=State)
    graph.add_node('factual_check', create_factual_check())
    graph.add_node("tools", create_search_tool_node())
    graph.add_node("keypoint_extract", create_keypoint_extract())

    graph.add_edge(START, "factual_check")
    graph.add_conditional_edges(
        "factual_check",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("keypoint_extract", "factual_check")
    return graph.compile(name="事实性检查")


if __name__ == '__main__':
    agent = create_factual_check_agent()

    for item in agent.stream({
        "messages": [
            "状元 球员分别是哪些",
        ],
    }, {
        "recursion_limit": 100
    }):
        print(item)
