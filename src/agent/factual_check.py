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
    search_results: Annotated[List[dict],add_messages]


llm = ChatTongyi(model="qwen-max-latest")

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
    你是一个问题回答助手，请根据用户的问题 使用 知识搜索工具进行检索，可以进行多次检索，并根据检索到的知识，确保能回答问题
    1. 你可以调用搜索工具进行知识检索
    2. 当前的检索结果中没有具体提及时，则需要进一步检索 缺失的信息
    3. 当搜索到的知识足够回答问题时，则直接给出回答。
    4. 仅回答与用户问题相关的信息
    当前日期为：2025年07月13号
    """
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{messages}"),
        ]
    ) | llm_with_tool

    def factual_check(_state: State) :
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
            "2021~2025届第一顺位球员分别是哪些",
        ],
    }):
        print(item)
