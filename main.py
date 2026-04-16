from typing import List

"""
Pydantic base model is going to give us a base class so we can inherit from in order to define a structured data scheme.

So it's going to provide functionality like data parsing and serialization and automatic type validations.

and the field class is going to allow us to add metadata to our models attribute.
"""
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()

"""
Purpose: Defines a simple schema for representing a "source" (e.g., a URL from a search result or API call). This ensures every source has a valid URL string.
Why needed: In your agent (which uses TavilySearch for web queries), sources are likely URLs from LinkedIn job postings. This model guarantees the data is structured and validated—e.g., it checks that url is a non-empty string. Without it, responses could be unstructured strings or dicts, leading to errors when parsing.
What it does:
Inherits from BaseModel, so Pydantic handles validation (e.g., raises an error if url is not a string).
The Field(description=...) adds metadata for documentation and potential use in APIs/tools.
Example instance: Source(url="https://linkedin.com/job/123") would be valid; Source(url=123) would fail validation.
"""

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

# Create LLM using Ollama
llm = ChatOllama(model="qwen2.5:7b")  # Assuming llama2 model, adjust as needed

# Define tools
tools = [TavilySearch()]

# Create the agent
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    
    # Invoke the agent with the query
    response = agent.invoke({"messages": [HumanMessage(content="search for 10 Firmware Engineer and Embedded engineer job openings In Hyderabad India with 2 to 3 years of experience using linkedin and return the linkedin links where I can click and apply. I want job opening links only")]})
    print("Agent response:", response)


if __name__ == "__main__":
    main()
