import json
from typing import TypedDict, List, Annotated
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from operator import add

dotenv.load_dotenv()


class State(TypedDict):
    content: str  # 用于存储员工信息和表现描述的原始文本内容
    tags: List[TypedDict]  # 用于存储待分析的标签定义，包括标签名称和标签定义规则
    task_description: str  # 用于存储任务描述
    search_tasks: List[TypedDict]
    search_results: Annotated[List, add]
    final_result: str


class SearchAndSummaryState(TypedDict):
    keyword: str
    tagName: str
    tagDefinePrompt: str
    searchResult: str
    summary: str


class SearchPlannerAgent:
    def __init__(self):
        user = """
        # 任务描述
        {taskDescription}
        
        ## 标签挖掘任务
        {tags}
        """
        system = """
        你的任务是 为下面的标签挖掘任务补充合适的业务知识，生成业务知识检索的query
        
        ## 字段说明
        1. tagName表示为某一个标签任务补充业务知识。
        2. keyword为需要检索业务知识的核心关键词实体，多个关键词 空格分隔
        
        ## 任务说明
        判断每一个标签是否需要补充业务知识。仅返回需要补充业务知识的标签名。
        
        仅输出如下json格式如下：
        [
            {{
               "tagName":"xxx",
               "keyword":"xxx"
            }}
        ]
        """
        model = ChatTongyi(model="qwen-max-latest")
        messages = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", user)
            ]
        )
        self.chain = (messages | model | JsonOutputParser())

    def invoke(self, _task_description, content, _tags):
        response = self.chain.invoke(
            {
                "taskDescription": _task_description,
                "content": content,
                "tags": _tags
            }
        )
        return {
            "search_tasks": response
        }


class SearchAndSummaryAgent:
    def __init__(self):
        user = """
        # 任务标签
        {tag}
        
        # 检索词
        {keyword}
        
        ## 检索结果
        {searchResult}
        """

        self.search_client = TavilySearchResults(
            max_results=10,
            description='tavily_search_results_json(query="the search query") - a search engine.',
        )

        system = """
        你的任务是根据任务标签、检索词 针对检索结果进行 筛选与任务标签和检索词相关联的内容、并总结成一条条核心业务知识点。
        1. 不能丢失关键信息。
        2. 仅返回知识点本身，不要输出其他信息
        3. 如果检索结果与内容无关，直接返回 “无有效内容”
        """
        model = ChatTongyi(model="qwen-max-latest")
        messages = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        self.chain = messages | model

    def invoke(self, search_task):
        search_result = self.search_client.invoke(input=search_task['keyword'])
        summary = self.chain.invoke(input={"searchResult": search_result,
                                           "tag": search_task['tagName'] + "\n" + search_task.get('tagDefinePrompt',
                                                                                                  ''),
                                           "keyword": search_task['keyword']}).content
        final_search_result = {"tagName": search_task['tagName'],
                               'searchResult': search_result,
                               'summary': summary}

        return {
            "search_results": [final_search_result]
        }


class AnalysisAgent:
    def __init__(self):
        user = """
        你是一个富有经验的数据分析专家，擅长从中挖掘各种信息。
        # 任务目标
        {taskDescription}
        
        # 待分析数据
        '''
        
        {content}
        
        '''
        
        # 注意事项
        - 仅输出结果本身，不输出推理、解析过程等
        
        # 标签体系
        {tags}
        
        # 输出格式
        {outputFormat}
        """
        model = ChatTongyi(model="qwen-max-latest")
        messages = ChatPromptTemplate.from_messages([("human", user)])
        self.chain = messages | model | JsonOutputParser()

    def invoke(self, _state: State):
        # 拼接提示词
        search_results = _state['search_results']
        # 按照 tagName生成 dict
        summary_dict = {item['tagName']: item['summary'] for item in search_results}

        final_tasks = ""

        output_format = {}

        for item in _state['tags']:
            tag_name = item['tagName']

            output_format['tagName'] = 'xxx'

            final_tasks += "标签名称：" + tag_name + "\n"
            final_tasks += "标签定义：" + item['tagDefinePrompt'] + "\n"
            summary = summary_dict.get(tag_name)
            if summary is not None:
                final_tasks += "可能参考的知识库：" + summary + "\n"
            final_tasks += "\n"

        final_result = self.chain.invoke({
            'content': _state['content'],
            'taskDescription': _state['task_description'],
            'tags': final_tasks,
            "outputFormat": "按照如下json返回。仅返回json本身\n"+json.dumps(output_format, ensure_ascii=False)
        })

        return {
            "final_result": final_result
        }


def create_search_planner_node():
    planner_agent = SearchPlannerAgent()
    return lambda _state: planner_agent.invoke(_state['task_description'], _state['content'], _state['tags'])


def create_search_summary_node():
    search_and_summary_agent = SearchAndSummaryAgent()
    return lambda _state: search_and_summary_agent.invoke(_state)


def create_analysis_node():
    search_and_summary_agent = SearchAndSummaryAgent()
    return lambda _state: search_and_summary_agent.invoke(_state)


def assign_workers(_state: State):
    return [Send("search_and_summary", _s) for _s in _state["search_tasks"]]


def create_analysis_agent():
    agent = AnalysisAgent()
    return lambda _state: agent.invoke(_state)


def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node('search_planner', create_search_planner_node())
    graph_builder.add_node("search_and_summary", create_search_summary_node())
    graph_builder.add_node("analysis_agent", create_analysis_agent())

    graph_builder.add_edge(START, "search_planner")
    graph_builder.add_conditional_edges("search_planner", assign_workers, ["search_and_summary"])
    graph_builder.add_edge("search_and_summary", "analysis_agent")
    graph_builder.add_edge("analysis_agent", END)
    # Compile the workflow
    return graph_builder.compile()


if __name__ == '__main__':
    content = """
        员工姓名：张三
        性别：男
        部门：市场部
        工作表现描述：在过去三个月中，张三超额完成了销售目标的120%，并带领团队成功实施了多个关键项目。他展现了卓越的领导能力和团队协作精神。
        """
    tags = [
        {
            "tagName": "是否违反员工守则",
            "tagDefinePrompt": "根据员工守则第一条判定，判断员工是否违反员工守则。0-否，1-是"
        },
        {
            "tagName": "员工等级",
            "tagDefinePrompt": "根据员工守则第五条判定，判断员工的等级。"
        },
        {
            "tagName": "员工性别",
            "tagDefinePrompt": "取值为男、女、未知"
        }
    ]
    task_description = "根据提供的员工信息和表现描述，分析并判定是否违反员工守则，同时评估员工等级。"
    state = {
        "content": content,
        "tags": tags,
        "task_description": task_description
    }
    # Invoke
    # final_state = compiled_workflow.invoke(state)
    # print(json.dumps(final_state, ensure_ascii=False))
    for s in build_graph().stream(state, stream_mode="values", subgraphs=True):
        print(json.dumps(s, ensure_ascii=False))
