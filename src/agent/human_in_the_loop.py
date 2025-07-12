from typing import TypedDict, Annotated, Sequence
from langgraph.graph.state import StateGraph
from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatTongyi
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.constants import START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages.tool import ToolMessage
import dotenv
dotenv.load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    name: str
    birthday: str


@tool
def human_assistant(query: str):
    """
    给出专业建议
    :param query:
    :return:
    """
    return interrupt({"query": query})

# 自定义状态更新
@tool
def human_assistance(
        name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # 获取第dict 第一个 KEY
    inner_human_response=  human_response.get(list(human_response.keys())[0])
    if inner_human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = inner_human_response.get("name", name)
        verified_birthday = inner_human_response.get("birthday", birthday)
        response = f"Made a correction: {inner_human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)

def chatbot():
    llm = ChatTongyi(model="qwen-max-latest")
    llm = llm.bind_tools([human_assistance])
    return lambda _state: {"messages": [llm.invoke(_state["messages"])]}


def create_graph():
    _graph = StateGraph(State)
    _graph.add_node("agent_node", chatbot())
    _graph.add_node("tools", ToolNode([human_assistance]))
    _graph.add_edge(START, "agent_node")
    _graph.add_conditional_edges(
        "agent_node",
        tools_condition,
        {"tools": "tools", END: END},
    )
    _graph.add_edge( "tools", "agent_node" )
    return _graph.compile()

if __name__ == '__main__':
    graph = create_graph()

    graph.checkpointer= MemorySaver()

    config = {"configurable": {"thread_id": "1"}}

    for item in graph.stream({"messages": ["I need some expert guidance for building an AI agent. Could you request assistance for me?"]}, config=config):
        print(item)

    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    for item in graph.stream(human_command,config=config):
        print(item)
