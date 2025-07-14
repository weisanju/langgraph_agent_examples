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
        你的任务是：根据下方的标签挖掘任务，判断每个标签是否需要补充相关的业务知识，并为需要补充的标签生成高质量、精准且全面的业务知识检索query。

        ## 输出要求
        - 仅输出那些确实需要补充业务知识的标签。
        - 对于每个需要补充的标签，需输出以下字段：
          1. tagName：需要补充业务知识的标签名称。

          2. keyword：
              2.1 :用于检索业务知识的核心关键词，要求精准、全面，多个关键词请用空格分隔，避免宽泛或无关词汇。
              2.2 :当 需要基于词库进行知识库召回时，需要使用正文内容检索，请直接返回  "{{content}}" 关键字
        
        ## 具体要求
        - 必须结合任务描述、标签定义和内容，综合分析判断哪些标签需要补充业务知识。
        - keyword字段应覆盖该标签业务知识检索的所有核心要素，确保检索结果相关且有用。
        - 输出仅为JSON数组，格式如下，不要输出任何多余内容或解释说明：
        [
            {{
                "tagName": "xxx",
                "keyword": "xxx"
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
            "outputFormat": "按照如下json返回。仅返回json本身\n" + json.dumps(output_format, ensure_ascii=False)
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
    # 判断 search_tasks 是否为空
    if _state["search_tasks"] is None or len(_state["search_tasks"]) == 0:
        return [Send("analysis_agent", _state)]
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


def state1():
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
    return {
        "content": content,
        "tags": tags,
        "task_description": task_description
    }


def state2():
    task_description = "对文本内容进行以下处理，纠错结果需要返回HTML格式，在原文上直接修改标记：错误文字用<span style='color:red'>标记，修改后文字用<span style='color:green'>标记"
    content = """
    近期，乌克兰总统泽连斯基与美国总统特朗普之间的 “隔空交锋” 不断升级，引发国际社会广饭关注。从相互指责到矿产协议争议，两人矛蹲持续激化。在 2 月 28 日美乌首脑会晤前夕，泽连斯基的公开表态更显焦灼，对特朗普提出三大核心质疑，直机美乌关系要害，深刻揭示出双方在战略利益、正红太地缘政制等方面的复杂博弈。
    """
    tags = [
        {
            "tagName": "标注后的内容",
            "tagDefinePrompt": """
                类型：敏感词标注,描述：错误文字标红，并且紧跟敏感词用括号包裹,标记语法：<span style='color:red'>错误的文字(敏感词)</span>, 敏感词列表需要 使用 待分析数据 原文生成关键字从知识库中检索
                类型：单词纠错标注,描述：错误文字标红，并且紧跟绿色修改后的文字，标记语法：<span style='color:red'>错误的文字</span><span style='color:green'>修改后文字</span>
                """
        }
    ]
    return {
        "content": content,
        "tags": tags,
        "task_description": task_description
    }


if __name__ == '__main__':

    # Invoke
    # final_state = compiled_workflow.invoke(state)
    # print(json.dumps(final_state, ensure_ascii=False))
    for s in build_graph().stream(state1(), stream_mode="values", subgraphs=True):
        print(json.dumps(s, ensure_ascii=False))
