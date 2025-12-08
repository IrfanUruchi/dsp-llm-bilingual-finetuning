import json
from bs4 import BeautifulSoup

def strip_html(x):
    return BeautifulSoup(x, "html.parser").get_text(separator=" ", strip=True)

inp = open("dsp_qa.jsonl", "r", encoding="utf-8")
out = open("dsp_qa_clean.jsonl", "w", encoding="utf-8")

for line in inp:
    item = json.loads(line)
    item["instruction"] = strip_html(item["instruction"])
    item["context"] = strip_html(item["context"])
    item["response"] = strip_html(item["response"])
    out.write(json.dumps(item, ensure_ascii=False) + "\n")

inp.close()
out.close()
