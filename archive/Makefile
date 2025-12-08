.PHONY: help gen verify eval train score report fmt
PY?=python

help:
	@echo "gen | verify | eval | train | score | report"

gen:
	@[ -n "$(TOPIC)" ] || (echo "TOPIC required"; exit 1)
	@[ -n "$(OUT)" ] || (echo "OUT required"; exit 1)
	$(PY) -m src.generators.$(TOPIC) --out $(OUT)

verify:
	@[ -n "$(IN)" ] || (echo "IN required"; exit 1)
	@[ -n "$(OUT)" ] || (echo "OUT required"; exit 1)
	$(PY) scripts/verify_and_promote.py --in $(IN) --out $(OUT) --cfg configs/dataset_v5.yaml

eval:
	@[ -n "$(TOPIC)" ] || (echo "TOPIC required"; exit 1)
	@[ -n "$(SUITE)" ] || (echo "SUITE required"; exit 1)
	@[ -n "$(OUT)" ] || (echo "OUT required"; exit 1)
	$(PY) -m src.eval.run_eval --topic $(TOPIC) --suite $(SUITE) --out $(OUT) --cfg configs/eval_v5.yaml

train:
	@[ -n "$(CFG)" ] || (echo "CFG required"; exit 1)
	$(PY) -m src.training.run_qlora --cfg $(CFG)

score:
	@[ -n "$(CFG)" ] || (echo "CFG required"; exit 1)
	@[ -n "$(OUT)" ] || (echo "OUT required"; exit 1)
	$(PY) -m src.training.score_model --cfg $(CFG) --out $(OUT)

report:
	@[ -n "$(IN)" ] || (echo "IN required"; exit 1)
	@[ -n "$(OUT)" ] || (echo "OUT required"; exit 1)
	$(PY) -m src.reports.build_report --in $(IN) --out $(OUT)

fmt:
	@ruff format || true

