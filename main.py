from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tavily import TavilyClient



load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    This is a tool that takes a query string as input and searches the internet.
    It prints the query to the console and returns a hardcoded string indicating
    that the weather in Tokyo is sunny.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)  # Replace with actual search logic if needed


# Create LLM using Ollama
llm = ChatOllama(model="qwen2.5:7b")  # Assuming llama2 model, adjust as needed

# Define tools
tools = [search]

# Create the agent
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    
    # Invoke the agent with the query
    response = agent.invoke({"messages": [HumanMessage(content="what's the weather in Tokyo")]})
    print("Agent response:", response)


if __name__ == "__main__":
    main()
