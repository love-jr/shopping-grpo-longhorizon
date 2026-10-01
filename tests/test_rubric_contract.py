import unittest

from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.rubric import (
    build_task_facts,
    extract_rubric_candidates,
    materialize_rubric_bundle,
)


class RubricContractTest(unittest.TestCase):
    def test_selected_constraint_requires_literal_query_evidence(self):
        query = "红色保温杯"
        facts = build_task_facts(
            task_id=7,
            query=query,
            target_product={"asin": "12345678", "category": "日用品›保温杯"},
            instruction_record={"instruction": query, "attributes": [], "instruction_options": []},
            reward_goal={"category": "日用品›保温杯"},
        )
        candidates = extract_rubric_candidates(facts)
        category = candidates["candidates"][0]
        response = {
            "selected_constraints": [{
                "candidate_id": category["candidate_id"],
                "description": "商品品类应为保温杯",
                "hardness": "hard",
                "query_quote": "保温杯",
                "selection_reason": "Query 明确要求保温杯",
            }]
        }

        bundle = materialize_rubric_bundle(
            task_facts=facts,
            candidates=candidates,
            curator_response=response,
            curator_model="deepseek-v4.1-flash",
            curator_prompt_version="rubric-curator-test",
            rubric_version="test",
        )
        self.assertEqual(bundle["rubrics"][0]["query_spans"][0]["text"], "保温杯")

        for quote in ("空调", "", "   "):
            with self.subTest(query_quote=quote):
                response["selected_constraints"][0]["query_quote"] = quote
                with self.assertRaises(ContractValidationError):
                    materialize_rubric_bundle(
                        task_facts=facts,
                        candidates=candidates,
                        curator_response=response,
                        curator_model="deepseek-v4.1-flash",
                        curator_prompt_version="rubric-curator-test",
                        rubric_version="test",
                    )

        bundle["rubrics"][0]["query_spans"] = []
        with self.assertRaises(ContractValidationError):
            validate_rubric_bundle(bundle)
        bundle["rubrics"] = []
        with self.assertRaises(ContractValidationError):
            validate_rubric_bundle(bundle)

    def test_empty_curator_selection_is_rejected(self):
        query = "保温杯"
        facts = build_task_facts(
            task_id=8,
            query=query,
            target_product={"asin": "12345678", "category": "保温杯"},
            instruction_record={"instruction": query, "attributes": [], "instruction_options": []},
            reward_goal={"category": "保温杯"},
        )
        candidates = extract_rubric_candidates(facts)
        with self.assertRaises(ContractValidationError):
            materialize_rubric_bundle(
                task_facts=facts,
                candidates=candidates,
                curator_response={"selected_constraints": []},
                curator_model="deepseek-v4.1-flash",
                curator_prompt_version="rubric-curator-test",
                rubric_version="test",
            )


if __name__ == "__main__":
    unittest.main()
