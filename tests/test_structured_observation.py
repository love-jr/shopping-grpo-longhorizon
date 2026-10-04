import unittest

from shopping_grpo.environment.actions import action_reject_reason, product_ids
from shopping_grpo.environment.observation import (
    StructuredObservationError,
    render_structured_observation,
)


def search_state(count=20):
    products = [
        {
            "rank": index,
            "asin": f"{index:012d}",
            "title": f"商品 {index}",
            "brand": "品牌",
            "category": "类目",
            "price": index,
            "key_attributes": ["属性"],
        }
        for index in range(1, count + 1)
    ]
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "search_available": False,
        "actions": ["back to search", "next >", *[p["asin"] for p in products]],
        "query": "商品",
        "normalized_query": "商品",
        "page": 1,
        "total_pages": 2,
        "total_results": 40,
        "rank_start": 1,
        "rank_end": count,
        "products": products,
    }


class StructuredObservationTest(unittest.TestCase):
    def test_all_twenty_products_are_visible_and_guard_actionable(self):
        visible = render_structured_observation(search_state())
        self.assertEqual(len(product_ids(visible)), 20)
        self.assertIsNone(
            action_reject_reason(
                "open_product",
                {"asin": "000000000020"},
                visible,
            )
        )

    def test_action_asin_mismatch_fails_closed(self):
        state = search_state(2)
        state["actions"].remove("000000000002")
        with self.assertRaisesRegex(StructuredObservationError, "actionable"):
            render_structured_observation(state)

    def test_hidden_reward_payload_is_rejected(self):
        state = search_state(1)
        state["reward"] = 1.0
        with self.assertRaisesRegex(StructuredObservationError, "forbidden"):
            render_structured_observation(state)

    def test_catalog_product_ids_from_eight_through_twelve_digits_are_supported(self):
        asins = ["12345678", "123456789", "1234567890", "35842622441", "123456789012"]
        state = search_state(len(asins))
        for product, asin in zip(state["products"], asins, strict=True):
            product["asin"] = asin
        state["actions"] = ["back to search", "next >", *asins]

        visible = render_structured_observation(state)

        self.assertEqual(product_ids(visible), asins)
        for asin in asins:
            self.assertIsNone(
                action_reject_reason("open_product", {"asin": asin}, visible)
            )

    def test_product_ids_outside_catalog_length_are_rejected(self):
        for invalid in ("1234567", "1234567890123"):
            with self.subTest(invalid=invalid):
                state = search_state(1)
                state["products"][0]["asin"] = invalid
                state["actions"] = ["back to search", "next >", invalid]
                with self.assertRaisesRegex(StructuredObservationError, "invalid"):
                    render_structured_observation(state)

    def test_search_values_cannot_shift_price_or_attribute_columns(self):
        state = search_state(1)
        product = state["products"][0]
        product.update({"brand": "A|B", "category": "杯|壶", "key_attributes": ["容量|500ml"], "title": "保温|杯"})
        row = next(line for line in render_structured_observation(state).splitlines() if line.startswith("1|"))
        self.assertEqual(row.split("|"), ["1", product["asin"], "1", "A｜B", "杯｜壶", "容量｜500ml", "保温｜杯"])

    def test_zero_price_is_not_rendered_as_missing(self):
        state = search_state(1)
        state["products"][0]["price"] = 0
        row = next(line for line in render_structured_observation(state).splitlines() if line.startswith("1|"))
        self.assertEqual(row.split("|")[2], "0")

    def test_information_subpage_list_content_is_readable_text(self):
        state = {
            "observation_version": "shopping-observation-v2",
            "page_type": "information_subpage",
            "search_available": False,
            "actions": ["< prev"],
            "product": {"asin": "12345678", "price": 0},
            "subpage": "features",
            "content": [" 保温12小时 ", "", "可机洗"],
        }
        observation = render_structured_observation(state)
        self.assertIn("content: 保温12小时; 可机洗", observation.splitlines())
        self.assertIn("price: 0", observation.splitlines())
        self.assertIsNone(action_reject_reason("prev_page", {}, observation))
        self.assertIsNotNone(action_reject_reason("select_option", {"value": "500ml"}, observation))


if __name__ == "__main__":
    unittest.main()
