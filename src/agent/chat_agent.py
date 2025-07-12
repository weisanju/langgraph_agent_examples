import json
from abc import ABC
from typing import TypedDict, List, Annotated, Sequence, Optional

import langgraph.constants
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_tavily.tavily_search import TavilySearch
from langchain_core.messages.tool import ToolMessage
from langgraph.constants import END
from langgraph.prebuilt.tool_node import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
import dotenv

dotenv.load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            args = tool_call["args"]
            print("Args", args)
            print("Tool call", tool_call["name"])
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                args
            )
            # print tool_result

            if 'error' in tool_result:
                content = str(tool_result)
            else:
                content = json.dumps(tool_result)

            outputs.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


search_tool = TavilySearch(max_results=2)


def chatbot():
    llm = ChatTongyi(model="qwen-max-latest")
    llm = llm.bind_tools([search_tool])
    return lambda _state: {"messages": [llm.invoke(_state["messages"])]}



def route_tools(
        state: State,
):
    """
    选择最后一条消息。作为AI消息。并且取其中的tool_calls作为消息
    :param state:
    :return:
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def create_chat_graph():
    chat_agent = StateGraph(state_schema=State)
    chat_agent.add_node("chatbot", chatbot())
    chat_agent.add_node("tools", ToolNode(tools=[search_tool]))
    chat_agent.add_edge(langgraph.constants.START, "chatbot")
    chat_agent.add_conditional_edges(
        "chatbot",
        tools_condition,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    chat_agent.add_edge("tools", "chatbot")

    return chat_agent.compile(name="对话机器人")


chat_graph = create_chat_graph()


def recursive_chat():
    chat_graph1  = create_chat_graph()
    chat_graph1.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        for event in chat_graph1.stream({"messages": [{"role": "user", "content": user_input}]},config=config):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
        snapshot = chat_graph1.get_state(config)
        print(snapshot)

if __name__ == '__main__':
    recursive_chat()
