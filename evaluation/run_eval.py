import json
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import sys
sys.path.append("../src")
from retrieval import retrieve_rerank

load_dotenv()
client = Anthropic()

# Load test set
with open("test_set.json", "r") as f:
    test_set = json.load(f)

results = []
for i, item in enumerate(test_set):
    # 1. Retrieve
    retrieved = retrieve_rerank(item["question"])
    
    # 2. Extract context texts into a list of strings
    contexts = [record["text"] for record in retrieved]
    
    # 3. Format contexts for the prompt
    context_block = ""
    for j, ctx in enumerate(contexts):
        context_block += f"<source_{j+1}>{ctx}</source_{j+1}>\n"
    
    # 4. Generate — single Claude call, no tool use
    response = client.messages.create(
        model= "claude-opus-4-8",
        max_tokens= 4096,
        system= "You are a Clinical research assistant, cite sources by PMID and title, flag uncertainty, say 'I don't know' if retrieved chunks don't contain the answer.",
        messages=[{
            "role": "user",
            "content": f"""Answer this clinical question using only the provided sources. Cite by source number.

            Question: {item['question']}

            Sources:
            {context_block}"""
        }]
    )
    
    answer = response.content[0].text
    
    # 5. Append to results
    results.append({
        "question": item["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": item["ground_truth"],
    })
    
    print(f"[{i+1}/{len(test_set)}] done")

# Save results
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)