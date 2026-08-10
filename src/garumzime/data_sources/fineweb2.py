from itertools import islice

from datasets import load_dataset

dataset = load_dataset("HuggingFaceFW/fineweb-2",
 name="lvs_Latn",
 split="train",
 streaming=True
 )

for row in islice(dataset, 1):
    print(row["text"][:200])



def donwload():
    