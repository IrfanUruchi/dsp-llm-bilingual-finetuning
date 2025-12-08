import json

input_path = "dsp_qa_clean.jsonl"
output_path = "dsp_qa_llama3.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)

        user_msg = item["instruction"]
        if item["context"].strip():
            user_msg += "\n\nContext:\n" + item["context"]

        record = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": item["response"]}
            ]
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Saved:", output_path)
