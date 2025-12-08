import json, yaml
from pathlib import Path

class BaseVerifier:
    def __init__(self, topic_name: str, config_dir: str = "config"):
        self.topic_name = topic_name
        self.paths = yaml.safe_load(open(f"{config_dir}/paths.yaml"))

    def verify(self, example: dict) -> bool:
        """To be implemented by subclass"""
        raise NotImplementedError

    def process_file(self, input_path):
        verified, rejected = [], []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if self.verify(ex):
                    ex["verified"] = True
                    verified.append(ex)
                else:
                    rejected.append(ex)
        out_dir = Path(self.paths["verified"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(input_path).name
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in verified:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return {
            "verified_count": len(verified),
            "rejected_count": len(rejected),
            "pass_rate": len(verified) / max(1, len(verified) + len(rejected))
        }
