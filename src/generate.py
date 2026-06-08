from dotenv import load_dotenv
from anthropic import Anthropic
from retrieval import retrieve_rerank
import json

load_dotenv()
client = Anthropic()
MODEL = "claude-opus-4-8"

tools = [
    {
        "name": "search_documents",
        "description": "Retrieve top 5 relevant documents with query from knowledge base",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    }
]

SYSTEM_PROMPT = "You are a Clinical research assistant, cite sources by PMID and title, flag uncertainty, say 'I don\'t know' if retrieved chunks don\'t contain the answer."


def run_agent(question: str, messages: list = None, facts: list = None) -> dict:
    """Run the full agentic RAG loop. Returns {"answer": str, "messages": list, "facts": list}"""
    if messages is None:
        messages = []
    if facts is None:
        facts = []

    messages.append({"role": "user", "content": question})

    # Context management
    if len(messages) > 20:
        facts.append(fact_extraction(messages))
        messages = messages[-20:]

    system = SYSTEM_PROMPT
    if facts:
        system += f" Facts from this conversation: {facts}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        tools=tools,
        system=system,
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=messages,
    )

    # Agentic loop
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = retrieve_rerank(block.input["query"])
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                    )
                except Exception as exc:
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True}
                    )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=tools,
            system=system,
            messages=messages,
        )

    final_text = next(block for block in response.content if block.type == "text").text
    messages.append({"role": "assistant", "content": final_text})

    return {"answer": final_text, "messages": messages, "facts": facts}


def fact_extraction(messages):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"You are a fact extraction assistant, extract facts from the following conversations. Identify and retain important facts. Do not change or edit any information. <message> {messages} </message>"
        }],
    )
    return response.content[0].text

if __name__ == "__main__":
    messages = []
    facts = []
    while True:
        user_input = input("User: ")
        if user_input in ["exit", "quit"]:
            break
        result = run_agent(user_input, messages, facts)
        messages = result["messages"]
        facts = result["facts"]
        print(result["answer"])