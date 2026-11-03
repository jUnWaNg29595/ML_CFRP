import json
import os

metrics_path = "/home/wangj/project/CFRP/extensions/ml_generalization_v2/reports/BASELINE_METRICS.json"
if not os.path.exists(metrics_path):
    print(f"Metrics file not found: {metrics_path}")
    exit(1)

with open(metrics_path, "r", encoding="utf-8") as f:
    d = json.load(f)

for task_k, task_v in d.items():
    target = task_v.get("target", "")
    unit = task_v.get("unit", "")
    print(f"\n--- {task_k} ({target}, {unit}) ---")
    print("Random 5-Fold (历史对照):")
    for m, res in task_v.get("random_5fold_metrics", {}).items():
        mae = res.get("Pooled_OOF_MAE", 0.0)
        r2 = res.get("Pooled_OOF_R2", 0.0)
        print(f"  {m:18s} | MAE: {mae:8.4f} | R2: {r2:8.4f}")
    
    print("Formulation Group 5-Fold (防同配方跨折泄漏):")
    for m, res in task_v.get("formulation_group_5fold_metrics", {}).items():
        mae = res.get("Pooled_OOF_MAE", 0.0)
        r2 = res.get("Pooled_OOF_R2", 0.0)
        print(f"  {m:18s} | MAE: {mae:8.4f} | R2: {r2:8.4f}")
    
    if "structure_family_metrics" in task_v:
        print("Structure Family Group (跨单体大类外推):")
        for m, res in task_v.get("structure_family_metrics", {}).items():
            mae = res.get("Pooled_OOF_MAE", 0.0)
            r2 = res.get("Pooled_OOF_R2", 0.0)
            print(f"  {m:18s} | MAE: {mae:8.4f} | R2: {r2:8.4f}")
