import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from ragas.llms import llm_factory

evaluator_llm = llm_factory("claude-opus-4-8", provider="anthropic")

# Load eval results
with open("eval_results.json", "r") as f:
    results = json.load(f)

dataset = Dataset.from_dict({
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "contexts": [r["contexts"] for r in results],
    "ground_truth": [r["ground_truth"] for r in results],
})

# Run evaluation
score = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=evaluator_llm
)

# Print results
print(score)

# Save to CSV 
df = score.to_pandas()
df.to_csv("eval_scores.csv", index=False)
print("\nSaved to eval_scores.csv")