from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to seach for
#     Return:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)
#     # return "Tokyo weather is sunny"


llm = ChatOpenAI(temperature=0, model="gpt-5")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What do you know about Sankhojjal Chatterjee?")}
    )
    print(result)


if __name__ == "__main__":
    main()
