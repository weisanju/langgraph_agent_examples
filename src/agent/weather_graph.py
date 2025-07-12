from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
import requests



def get_weather(city: str) -> str:
    """
    参数说明：
        city: 城市名称
    """
    print(f"获取{city}的天气信息...")
    try:
        # 构建 wttr.in API 的参数
        url = f"https://wttr.in/{city}"
        params = {
            "lang": "zh",
        }
        # 移除值为 None 的参数
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            return f"无法获取{city}的天气信息，API返回状态码: {response.status_code}"
    except Exception as e:
        return f"获取天气信息时出错: {e}"


weather_graph = create_react_agent(
    model=ChatTongyi(model="qwen-max-latest"),
    tools=[get_weather],
    prompt="You are a helpful assistant",
    name="天气助手"
)
