from Bio import Entrez, Medline
import json
from pathlib import Path

Entrez.email = "trinhvo00@gmail.com"

handle = Entrez.esearch(db="pubmed", retmax=500, term="Type 2 diabetes", idtype="acc")
records = Entrez.read(handle)
fetch_handle = Entrez.efetch(db="pubmed", id=records["IdList"], rettype="medline", retmode="text")

fetched_records = Medline.parse(fetch_handle)
for record in fetched_records:
    # Save results
    output_path = Path(f"../data/{record['PMID']}.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f)
    
    print(f"\nResults saved to {output_path}")
