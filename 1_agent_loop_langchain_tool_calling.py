from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen2.5:7b"


# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)  # 0 is default value returned if product not found


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---

"""
@traceable(name="LangChain Agent Loop") is a decorator from LangSmith (imported as from langsmith import traceable).
Purpose: It enables tracing and logging for the decorated function (run_agent). LangSmith tracks executions, inputs/outputs, 
and performance metrics (e.g., latency, errors) for debugging, monitoring, and analytics in production LangChain apps.
Why needed?: Without it, you can't observe agent behavior in LangSmith's dashboard. It's optional but useful for complex 
agents to log iterations, tool calls, and results.
What it does: Wraps the function to send data to LangSmith on each run. The name parameter labels the trace for easy 
identification.
"""

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # Main loop: iterate up to MAX_ITERATIONS times to process agent reasoning and tool calls
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Step 1: Invoke the LLM with the current message history
        # The LLM will decide whether to use tools or provide a final answer
        ai_message = llm_with_tools.invoke(messages)

        # Step 2: Extract any tool calls requested by the LLM
        # tool_calls is a list of tool invocations the LLM wants to execute. It is a list
        tool_calls = ai_message.tool_calls

        # Step 3: Check if the LLM provided a direct answer without using tools
        # If no tool calls were made, the LLM has completed its reasoning
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content  # Exit the loop and return the final answer. final ans is in content of ai_message

        # Step 4: Extract details from the FIRST tool call only (enforced: one tool per iteration)
        # This design prevents multiple tool calls per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")  # The name of the tool to invoke (e.g., "get_product_price")
        tool_args = tool_call.get("args", {})  # Arguments to pass to the tool (empty dict if none)
        tool_call_id = tool_call.get("id")  # Unique ID for this tool call (needed to link results back)

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Step 5: Look up the tool function from the tools dictionary using its name
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")  # Safety check: fail if tool doesn't exist

        # Step 6: Execute the tool with the provided arguments
        # The observation is the tool's return value (e.g., the product price)
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # Step 7: Append the assistant's message and tool result to the conversation history
        # This builds a complete dialogue that the LLM can reference in the next iteration
        messages.append(ai_message)  # Add the assistant's reasoning and tool request
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)  # Add the tool's response
        )  # These messages are then passed back to the LLM for the next iteration
        #print(messages)
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")