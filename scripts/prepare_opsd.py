#!/usr/bin/env python3
"""Build teacher-only contracts from the frozen catalog using the ShopSimulator Python."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/grpo/train.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/opsd-core/contracts.json",
        help="必须与 configs/opsd.yaml 的 shopping_opsd.contracts_path 一致",
    )
    args = parser.parse_args()
    task_ids = {json.loads(line)["task_id"] for line in args.tasks.read_text().splitlines()}
    held_out = {
        json.loads(line)["task_id"]
        for line in (ROOT / "data/evaluation/tasks.jsonl").read_text().splitlines()
    }
    if task_ids & held_out:
        raise ValueError("OPSD training tasks overlap the evaluation set")

    sys.path.insert(0, str(ROOT / "environments/ShopSimulator/shop_env"))
    sys.path.insert(0, str(ROOT / "src"))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    from web_agent_site.utils import DEFAULT_FILE_PATH
    from web_agent_site.engine.variant_price import (
        matching_candidate_options,
        resolve_variant_price,
    )
    from shopping_grpo.training.opsd.privilege import purchase_contract, reference_rejection_reasons

    products, by_asin, prices, _ = load_products(DEFAULT_FILE_PATH)
    goals = get_goals(products, prices)
    manifest = json.loads((ROOT / "data/environment.json").read_text())

    def contract(task_id):
        goal = goals[task_id]
        product = by_asin[goal["asin"]]
        selected, option_check = matching_candidate_options(
            product, goal["required_options_by_key"]
        )
        if option_check["status"] != "pass":
            selected = {}
        price = resolve_variant_price(product, selected)
        reference = {
            "selected_options": selected,
            "variant_price": price["price"] if price["status"] == "pass" else None,
        }
        return purchase_contract(goal, product, reference)
    contracts = {}
    for task_id in sorted(task_ids):
        value = contract(task_id)
        reasons = reference_rejection_reasons(value)
        if reasons:
            raise ValueError(f"OPSD task {task_id} has an unusable reference: {'; '.join(reasons)}")
        contracts[str(task_id)] = value
    if not contracts:
        raise ValueError("OPSD requires nonempty task references")
    payload = {
        "product_data_sha256": manifest["product_data_sha256"],
        "contracts": contracts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(contracts)} teacher references")


if __name__ == "__main__":
    main()
