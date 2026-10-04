"""按 token 预算裁剪 observation v2 的标题/正文，保留完整决策字段和动作 footer。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from shopping_grpo.environment.actions import clickable_buttons, product_ids
from shopping_grpo.environment.observation import HEADER
from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE


TRUNCATION_MARKER = "[TRUNCATED_BY_SHOPPING_PROJECTOR]"
FOOTER_PATTERN = re.compile(
    r"\n\n搜索功能是否可用: (True|False)\n可点击的按钮: (\[[^\n]*\])\Z"
)


class ObservationProjectionError(RuntimeError):
    """The complete decision fields and actionable state cannot fit the budget."""


@dataclass(frozen=True)
class ObservationProjectionMeta:
    tool_name: str
    page_type: str
    token_budget: int
    raw_tokens: int
    visible_tokens: int
    truncation_ratio: float
    truncated: bool
    raw_asin_count: int
    visible_asin_count: int
    raw_button_count: int
    visible_button_count: int
    critical_footer_preserved: bool

    def to_dict(self):
        return asdict(self)


def project_observation(
    tool_name,
    observation,
    *,
    count_tokens,
    token_budget=4096,
    detail_token_budget=4096,
    generic_token_budget=768,
    search_top_k=20,
):
    """只裁标题和信息子页正文；关键字段装不下时停止，而不是改写价格/规格。"""
    observation = str(observation)
    token_budget = int(token_budget)
    detail_token_budget = int(detail_token_budget)
    generic_token_budget = int(generic_token_budget)
    search_top_k = int(search_top_k)
    if min(token_budget, detail_token_budget, generic_token_budget) < 64:
        raise ValueError("all observation token budgets must be at least 64")
    if search_top_k < 1:
        raise ValueError("search_top_k must be positive")

    lines = observation.splitlines()
    page_type = (
        lines[1].removeprefix("page_type: ")
        if len(lines) > 1 and lines[0] == HEADER and lines[1].startswith("page_type: ")
        else "generic"
    )
    effective_budget = {
        "search_results": token_budget,
        "product_detail": detail_token_budget,
        "information_subpage": detail_token_budget,
    }.get(page_type, generic_token_budget)
    raw_buttons = clickable_buttons(observation)
    raw_asins = product_ids(observation)
    if page_type == "search_results" and len(raw_asins) > search_top_k:
        raise ObservationProjectionError("search page exceeds configured page capacity")

    raw_tokens = int(count_tokens(observation))
    visible = observation
    if raw_tokens > effective_budget:
        footer = FOOTER_PATTERN.search(observation)
        if footer is None:
            raise ObservationProjectionError("long observation has no complete action footer")
        if not lines or lines[0] != HEADER:
            raise ObservationProjectionError("projection requires observation v2")
        body_lines = observation[:footer.start()].splitlines()
        flexible = {}
        for index, line in enumerate(body_lines):
            if page_type == "search_results" and re.match(
                rf"^\d+\|{PRODUCT_ID_CAPTURE}\|", line
            ):
                fields = line.split("|", 6)
                if len(fields) != 7:
                    raise ObservationProjectionError("malformed search product row")
                flexible[index] = ("|".join(fields[:6]) + "|", fields[6])
            elif page_type in {"product_detail", "information_subpage"} and line.startswith(
                ("title: ", "content: ")
            ):
                prefix, text = line.split(": ", 1)
                flexible[index] = (prefix + ": ", text)

        def render(character_limit):
            projected = list(body_lines)
            for index, (prefix, text) in flexible.items():
                projected[index] = prefix + _compact_text(text, character_limit)
            return "\n".join([*projected, TRUNCATION_MARKER]) + footer.group(0)

        # 空标题/正文是硬字段的下界；下界超预算时绝不能把价格也裁掉。
        visible = render(0)
        if int(count_tokens(visible)) > effective_budget:
            raise ObservationProjectionError(
                "complete decision fields and action footer exceed observation token budget"
            )
        low, high = 1, max((len(text) for _, text in flexible.values()), default=0)
        while low <= high:
            middle = (low + high) // 2
            candidate = render(middle)
            if int(count_tokens(candidate)) <= effective_budget:
                visible = candidate
                low = middle + 1
            else:
                high = middle - 1

    visible_tokens = int(count_tokens(visible))
    visible_buttons = clickable_buttons(visible)
    visible_asins = product_ids(visible)
    if visible_tokens > effective_budget:
        raise ObservationProjectionError("projected observation exceeds token budget")
    if visible_buttons != raw_buttons or visible_asins != raw_asins:
        raise ObservationProjectionError("projection changed actionable targets")
    if page_type == "search_results" and set(raw_asins) != {
        button for button in raw_buttons if re.fullmatch(PRODUCT_ID_CAPTURE, button)
    }:
        raise ObservationProjectionError("search products differ from actionable targets")

    return visible, ObservationProjectionMeta(
        tool_name=str(tool_name),
        page_type=page_type,
        token_budget=effective_budget,
        raw_tokens=raw_tokens,
        visible_tokens=visible_tokens,
        truncation_ratio=(visible_tokens / raw_tokens if raw_tokens else 1.0),
        truncated=visible != observation,
        raw_asin_count=len(raw_asins),
        visible_asin_count=len(visible_asins),
        raw_button_count=len(raw_buttons),
        visible_button_count=len(visible_buttons),
        critical_footer_preserved=True,
    )


def _compact_text(text, character_limit):
    if len(text) <= character_limit:
        return text
    if character_limit <= 0:
        return ""
    if character_limit == 1:
        return "…"
    head = (character_limit - 1) * 2 // 3
    tail = character_limit - 1 - head
    return text[:head] + "…" + (text[-tail:] if tail else "")
