# 事实性检查Agent

from operator import add
from typing import Annotated, TypedDict, List, Literal
from langchain_tavily import TavilySearch

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.types import Send, Command

from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    search_results: Annotated[List[dict], add_messages]


llm = ChatTongyi(model="qwen-max-latest", model_kwargs={
    "enable_thinking": False
})

_search_tool = TavilySearch(max_results=5)


def create_search_tool_node():
    def search(state: State) -> Command[Literal["keypoint_extract"]]:
        # 获取最后一个
        tool_call = state["messages"][-1].tool_calls[0]
        args = tool_call['args']
        search_results = _search_tool.invoke(tool_call)
        return Command(
            update={
                "search_results": search_results,
                "tool_call_id": tool_call['id'],
            },
            goto=Send('keypoint_extract', {
                "query": args['query'],
                "tool_call_id": tool_call['id'],
                "search_results": search_results.content,
            }))

    return search


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
    llm_with_tool = llm.bind_tools([_search_tool])
    system = """
    当前日期为：2025年07月13号
    
    你是一个问题回答助手，请根据用户的问题，使用知识搜索工具进行检索。你可以多次调用检索工具，直到获取到足够的信息来回答用户的问题。请遵循以下要求：

    1. 首先，分析用户的问题，确定需要哪些关键信息来完整回答。
    2. 使用搜索工具进行知识检索，获取相关信息。
    3. 如果当前检索结果中存在信息缺失、内容不具体或无法直接回答用户问题时，请明确指出缺失的具体信息，并针对这些缺失点，进一步调用检索工具进行补充检索，直到所有关键信息都被补全。
    4. 检索过程中，每次都要判断现有信息是否足以回答问题，若不足，请继续主动检索缺失的信息内容。
    5. 仅回答与用户问题直接相关的信息，避免无关内容。
    6. 当你认为已经获得了足够且准确的知识来回答用户问题时，请对所有检索到的结果进行归纳总结，给出一个清晰、合理的最终回答。
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
            "2002~2025届第一顺位球员分别是哪些",
        ],
    }, {
        "recursion_limit": 40
    }):
        print(item)
