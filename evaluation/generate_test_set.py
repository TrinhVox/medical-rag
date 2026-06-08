import json, random
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# Load 30 random abstracts from data/
data_folder = Path("../data")
all_files = list(data_folder.iterdir())
sample_files = random.sample(all_files, min(30, len(all_files)))

test_set = []
for file in sample_files:
    with open(file, "r") as f:
        record = json.load(f)
    if not record.get("AB"):
        continue
    
    # Ask Claude to generate a question + answer from the abstract
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Given this PubMed abstract, generate one specific clinical question that this abstract answers, and provide the answer based only on the abstract.

                    Title: {record['TI']}
                    Abstract: {record['AB']}

                    Respond in this exact JSON format only, no other text:
                    {{"question": "...", "ground_truth": "..."}}"""
        }]
    )
    
    # Parse answers
    try:
        qa = json.loads(response.content[0].text)
        qa["pmid"] = record["PMID"]
        test_set.append(qa)
        print(f"[{len(test_set)}] {qa['question'][:80]}...")
    except json.JSONDecodeError:
        print(f"Skipping {record['PMID']} — couldn't parse JSON")
        continue



# Save
with open("test_set.json", "w") as f:
    json.dump(test_set, f, indent=2)
print(f"\nSaved {len(test_set)} test questions")