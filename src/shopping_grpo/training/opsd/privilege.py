"""Teacher-only task facts; never an observation or an executable action."""

import json


TEACHER_INSTRUCTION = (
    "\n\n[教师购买参考]\n用户原始需求是最终依据；源 instruction_options 只是标注，"
    "可能不完整或与原话冲突，不能覆盖用户 Query；预算解析为空不代表没有预算。"
    "参考商品用于帮助选对商品，不是已核验的购买答案。只提供明确匹配的规格，"
    "未指定规格不需要补选；未解析标注仍需结合原话判断，价格为空表示无法确定。"
    "店铺不是品牌，目录类目不是新增需求。"
    "可用参考标题和规格关键词搜索，但参考信息不是当前页面的观察证据，"
    "只能操作最新 observation 中可见的商品和选项。"
    "沿学生的真实工具历史继续；同轴选新值会替换旧值，购买前复核最终已选规格"
    "与实际价格，保留已经满足的要求。没有合适商品时继续探索或合理结束。"
    "不要复述参考答案；照常输出一个合法购物工具调用。\n"
)


def purchase_contract(goal: dict, product: dict, reference: dict) -> dict:
    """Expose the catalog product reference to the teacher, never identifiers or reward."""
    return {
        "user_request": goal["instruction_text"],
        "annotations": {
            "attributes": goal["attributes"],
            "option_values": goal["goal_options"],
            "unresolved_options": goal["unresolved_option_requirements"],
            "parsed_budget_upper": goal["price_upper"],
        },
        "reference_purchase": {
            "title": product["Title"],
            "shop_name": product.get("shop_name", ""),
            "catalog_category": product["category"],
            "attributes": product.get("Attributes") or product.get("attribute") or [],
            "available_options": product.get("options") or {},
            "selected_options": reference["selected_options"],
            "variant_price": reference["variant_price"],
        },
    }


def reference_rejection_reasons(contract: dict) -> list[str]:
    """Require only enough reference text to guide the teacher."""
    if not isinstance(contract, dict):
        return ["contract must be an object"]
    purchase = contract.get("reference_purchase")
    purchase = purchase if isinstance(purchase, dict) else {}
    reasons = []
    request = contract.get("user_request")
    if not isinstance(request, str) or not request.strip():
        reasons.append("user_request must be nonempty text")
    title = purchase.get("title")
    if not isinstance(title, str) or not title.strip():
        reasons.append("reference_purchase.title must be nonempty text")
    return reasons


def teacher_messages(messages: list[dict], contract: dict) -> list[dict]:
    """Copy the student messages and extend only the teacher's system context."""
    result = [dict(message) for message in messages]
    hint = TEACHER_INSTRUCTION + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
    result[0]["content"] += hint
    return result
