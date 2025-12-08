import json

REQUIRED_KEYS = ["id","lang_pair","topic","subtopic",
                 "prompt_en","response_en","prompt_sq","response_sq",
                 "meta"]

def check_schema(row: dict):
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"Missing key: {k}")
    json.dumps(row)  
