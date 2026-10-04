"""投影只裁文本，不改价格、规格、当前页候选和动作。"""

import unittest

from shopping_grpo.environment.actions import action_reject_reason, clickable_buttons, product_ids
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.projection import ObservationProjectionError, project_observation


def search_state(count=20):
    products = [
        {
            "rank": index,
            "asin": f"{index:012d}",
            "price": "12345.67",
            "brand": "品牌",
            "category": "保温杯",
            "key_attributes": ["不锈钢", "500ml"],
            "title": "超长商品标题" * 100,
        }
        for index in range(1, count + 1)
    ]
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "search_available": False,
        "actions": ["back to search", "next >", *[p["asin"] for p in products]],
        "query": "保温杯",
        "normalized_query": "保温杯",
        "page": 2,
        "total_pages": 2,
        "total_results": 40,
        "rank_start": 21,
        "rank_end": 40,
        "products": products,
    }


def product_state(page_type="product_detail"):
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": page_type,
        "search_available": False,
        "actions": ["back to search", "buy now", "500ml"],
        "product": {
            "asin": "12345678",
            "price": 99.99,
            "brand": "品牌",
            "category": "保温杯",
            "key_attributes": ["不锈钢"],
            "title": "超长商品标题" * 100,
        },
        "selected_options": {"容量": "500ml"},
        "available_options": {"容量": ["500ml"]},
        "subpage": "description",
        "content": "长描述" * 100 + "尾部规格",
    }


class ObservationProjectionTest(unittest.TestCase):
    def test_search_keeps_all_decision_fields_and_last_actionable_product(self):
        state = search_state()
        raw = render_structured_observation(state)
        visible, meta = project_observation(
            "next_page", raw, count_tokens=len, token_budget=1800
        )
        original_rows = [line.split("|", 6) for line in raw.splitlines() if line[:1].isdigit()]
        visible_rows = [line.split("|", 6) for line in visible.splitlines() if line[:1].isdigit()]
        self.assertEqual([r[:6] for r in original_rows], [r[:6] for r in visible_rows])
        self.assertIn("Page 2 of 2", visible)
        self.assertEqual(product_ids(visible), product_ids(raw))
        self.assertEqual(clickable_buttons(visible), state["actions"])
        self.assertIsNone(action_reject_reason("open_product", {"asin": state["products"][-1]["asin"]}, visible))
        self.assertLessEqual(len(visible), 1800)
        self.assertTrue(meta.truncated)

    def test_tight_budget_fails_instead_of_corrupting_price(self):
        raw = render_structured_observation(search_state(1))
        with self.assertRaisesRegex(ObservationProjectionError, "decision fields"):
            project_observation("search_products", raw, count_tokens=len, token_budget=320)

    def test_product_and_subpage_keep_price_options_and_page_identity(self):
        for page_type in ("product_detail", "information_subpage"):
            with self.subTest(page_type=page_type):
                state = product_state(page_type)
                raw = render_structured_observation(state)
                visible, meta = project_observation(
                    "open_product", raw, count_tokens=len,
                    detail_token_budget=500, generic_token_budget=128,
                )
                for line in raw.splitlines():
                    if line.startswith(("asin:", "price:", "brand:", "category:", "key_attributes:", "selected_options:", "available_options:", "subpage:")):
                        self.assertIn(line, visible.splitlines())
                self.assertEqual(meta.page_type, page_type)
                self.assertEqual(clickable_buttons(visible), state["actions"])
                self.assertLessEqual(len(visible), 500)
                self.assertTrue(meta.truncated)

    def test_untruncated_product_is_returned_unchanged(self):
        state = product_state()
        state["product"]["title"] = "保温杯"
        raw = render_structured_observation(state)
        visible, meta = project_observation("open_product", raw, count_tokens=len)
        self.assertEqual(visible, raw)
        self.assertFalse(meta.truncated)

    def test_page_capacity_is_checked_even_below_token_budget(self):
        raw = render_structured_observation(search_state(2))
        with self.assertRaisesRegex(ObservationProjectionError, "page capacity"):
            project_observation("search_products", raw, count_tokens=len, token_budget=10000, search_top_k=1)

    def test_unknown_long_text_is_not_silently_projected_as_a_product(self):
        with self.assertRaises(ObservationProjectionError):
            project_observation("unknown", "x" * 1000, count_tokens=len, generic_token_budget=128)


if __name__ == "__main__":
    unittest.main()
