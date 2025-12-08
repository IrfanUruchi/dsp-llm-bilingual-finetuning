import xml.etree.ElementTree as ET
import json

INPUT_FILE = "Posts.xml"
OUTPUT_FILE = "dsp_qa.jsonl"

tree = ET.parse(INPUT_FILE)
root = tree.getroot()

questions = {}
answers = {}

for row in root.findall("row"):
    post_type = row.get("PostTypeId")
    post_id = row.get("Id")

    if post_type == "1": 
        questions[post_id] = {
            "title": row.get("Title"),
            "body": row.get("Body"),
            "accepted": row.get("AcceptedAnswerId")
        }

    elif post_type == "2":  
        parent = row.get("ParentId")
        if parent not in answers:
            answers[parent] = []
        answers[parent].append({
            "id": row.get("Id"),
            "body": row.get("Body")
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for qid, q in questions.items():
        acc_id = q["accepted"]
        if not acc_id:
            continue
        if qid not in answers:
            continue

        accepted = None
        for ans in answers[qid]:
            if ans["id"] == acc_id:
                accepted = ans["body"]
                break
        
        if not accepted:
            continue
        
        record = {
            "instruction": q["title"] or "",
            "context": q["body"] or "",
            "response": accepted or ""
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Finished! Saved to", OUTPUT_FILE)
