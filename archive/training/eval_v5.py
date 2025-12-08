import yaml, json
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate(cfg_path="training/configs/eval_v5.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    model = AutoModelForCausalLM.from_pretrained(cfg["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    eval_data = [json.loads(l) for l in open(cfg["eval_set"], "r")]
    total, correct = 0, 0

    for ex in eval_data:
        inputs = tokenizer(f"Q: {ex['prompt_en']}", return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=200)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        if ex["expected_keyword"].lower() in text.lower():
            correct += 1
        total += 1

    print(f"Accuracy: {correct/total:.2%}")

if __name__ == "__main__":
    evaluate()
