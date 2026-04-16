from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()


# Create LLM using Ollama
llm = ChatOllama(model="qwen2.5:7b")  # Assuming llama2 model, adjust as needed

# Define tools
tools = [TavilySearch()]

# Create the agent
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    
    # Invoke the agent with the query
    response = agent.invoke({"messages": [HumanMessage(content="search for 10 Firmware Engineer and Embedded engineer job openings In Hyderabad India with 2 to 3 years of experience using linkedin and return the linkedin links where I can click and apply.")]})
    print("Agent response:", response)


if __name__ == "__main__":
    main()
