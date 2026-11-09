"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""Bạn là VinAssistant, trợ lý AI của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: Tư vấn sản phẩm, dịch vụ và tiếp nhận yêu cầu hỗ trợ của Vingroup.
- Phong cách: Chuyên nghiệp, thân thiện, ngắn gọn và chính xác; trả lời bằng ngôn ngữ của khách hàng.

## AVAILABLE TOOLS
- search_product_catalog: Tra cứu sản phẩm theo danh mục và giá tối đa.
- submit_support_ticket: Tạo yêu cầu hỗ trợ, ghi nhận tên khách hàng và mức độ ưu tiên.
Tool schemas: {json.dumps(TOOL_DEFINITIONS, ensure_ascii=False)}

## CORE RULES
1. Không bịa thông tin, giá, chính sách, tình trạng hàng hoặc kết quả thao tác.
2. Phải dùng search_product_catalog khi cần tra cứu hay so sánh sản phẩm/dịch vụ.
3. Phải dùng submit_support_ticket khi khách yêu cầu ghi nhận sự cố hoặc yêu cầu hỗ trợ.
4. Chỉ khẳng định thông tin có trong kết quả tool hoặc dữ liệu đã được cung cấp; nêu rõ khi chưa đủ dữ liệu.
5. Không tự suy đoán thông tin bắt buộc còn thiếu để gọi tool; hãy hỏi khách hàng.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ sản phẩm, dịch vụ và yêu cầu thuộc hệ sinh thái Vingroup.
- Từ chối lịch sự yêu cầu ngoài phạm vi này.

## OUTPUT CONTRACT
- Ghi lại từng bước theo thứ tự Thought → Action → Observation.
- Kết thúc bằng Final Answer, trình bày rõ kết quả hoặc thông tin còn thiếu.
"""


def _normalize_text(text: str) -> str:
    """Remove Vietnamese accents for robust keyword matching."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)


def _extract_max_price(text: str) -> int:
    normalized = _normalize_text(text)
    amount_match = re.search(
        r"(\d[\d.,]*)\s*(ty|trieu|tr|nghin|ngan|vnd|dong)\b",
        normalized,
    )
    if amount_match:
        amount_text, unit = amount_match.groups()
        if "," in amount_text or "." in amount_text:
            parts = re.split(r"[.,]", amount_text)
            if len(parts[-1]) <= 2:
                amount = float(amount_text.replace(",", "."))
            else:
                amount = int("".join(parts))
        else:
            amount = int(amount_text)

        multiplier = {
            "ty": 1_000_000_000,
            "trieu": 1_000_000,
            "tr": 1_000_000,
            "nghin": 1_000,
            "ngan": 1_000,
            "vnd": 1,
            "dong": 1,
        }[unit]
        return int(amount * multiplier)

    amount_match = re.search(
        r"(?:duoi|gia|toi da|ngan sach)\s*(?:duoi\s*)?(\d[\d.,]*)",
        normalized,
    )
    if amount_match:
        return int("".join(character for character in amount_match.group(1) if character.isdigit()))
    return 999999999999


def _extract_customer_name(text: str) -> Tuple[Optional[str], Optional[re.Match]]:
    name_pattern = re.compile(
        r"(?:(?i:tôi\s+tên\s+là|tên\s+tôi\s+là|tôi\s+tên|tên\s+là))\s+"
        r"([A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]*(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]*){0,4})"
    )
    match = name_pattern.search(text)
    if not match:
        return None, None
    return match.group(1).strip(), match


def _extract_ticket_issue(text: str, name_match: Optional[re.Match]) -> str:
    issue = text[name_match.end():] if name_match else text
    issue = re.split(r"[.!?]", issue, maxsplit=1)[0]
    return issue.strip(" \t\r\n,;:-")


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        normalized = _normalize_text(user_input)
        has_vingroup_reference = any(
            keyword in normalized
            for keyword in (
                "vinfast", "xe dien", "vinpearl", "du lich", "resort",
                "khach san", "vingroup",
            )
        )
        has_catalog_request = any(
            keyword in normalized
            for keyword in (
                "gia", "duoi", "toi da", "bao nhieu tien", "xem", "tim",
                "goi y", "mua", "san pham", "resort", "du lich", "khach san",
            )
        )
        needs_catalog = has_vingroup_reference and has_catalog_request
        needs_ticket = _contains_any(
            normalized,
            (
                "ghi nhan", "tao ticket", "ho tro", "phan hoi", "phan anh",
                "khieu nai", "loi", "hong", "su co",
            ),
        )

        customer_name, name_match = _extract_customer_name(user_input)
        ticket_issue = _extract_ticket_issue(user_input, name_match)
        needs_customer_name = needs_ticket and not customer_name
        priority = "medium"
        if _contains_any(normalized, ("nghiem trong", "khan cap", "gap", "urgent")):
            priority = "high"
        elif _contains_any(normalized, ("muc do thap", "uu tien thap", "it nghiem trong")):
            priority = "low"

        actions: List[Tuple[str, Dict[str, Any]]] = []
        if needs_catalog:
            category = (
                "du_lich"
                if any(keyword in normalized for keyword in ("vinpearl", "du lich", "resort", "khach san"))
                else "xe_dien"
            )
            actions.append((
                "search_product_catalog",
                {"category": category, "max_price": _extract_max_price(user_input)},
            ))
        if needs_ticket and customer_name:
            actions.append((
                "submit_support_ticket",
                {
                    "customer_name": customer_name,
                    "issue_description": ticket_issue or user_input.strip(),
                    "priority": priority,
                },
            ))

        self.trace.append({
            "step": "thought",
            "content": {
                "needs_catalog": needs_catalog,
                "needs_ticket": needs_ticket,
            },
        })
        if needs_customer_name:
            self.trace.append({
                "step": "observation",
                "content": "Cần tên khách hàng trước khi tạo yêu cầu hỗ trợ.",
            })

        if self.max_iterations < 1:
            answer = "Lỗi: Vượt quá số bước tối đa."
            self.trace.append({"step": "final_answer", "content": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached",
            }

        observations: Dict[str, Any] = {}
        iteration = 0
        for tool_name, arguments in actions:
            if iteration >= self.max_iterations:
                answer = "Lỗi: Vượt quá số bước tối đa trước khi hoàn tất yêu cầu."
                self.trace.append({"step": "final_answer", "content": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached",
                }

            iteration += 1
            self.trace.append({
                "step": "action",
                "iteration": iteration,
                "tool": tool_name,
                "arguments": arguments,
            })
            observation = TOOL_MAP[tool_name](**arguments)
            observations[tool_name] = observation
            self.trace.append({
                "step": "observation",
                "iteration": iteration,
                "tool": tool_name,
                "result": observation,
            })

        if not actions:
            iteration = 1
            if needs_customer_name:
                answer = "Để ghi nhận yêu cầu hỗ trợ, vui lòng cho tôi biết tên khách hàng."
            elif "bao hanh" in normalized and "pin" in normalized and "xe dien" in normalized:
                answer = (
                    "Theo dữ liệu sản phẩm hiện có, VinFast VF 5 Plus được ghi nhận "
                    "bảo hành pin 10 năm. Dữ liệu này chưa xác nhận chính sách cho mọi mẫu xe."
                )
            elif has_vingroup_reference or "vingroup" in normalized:
                answer = (
                    "Tôi chưa có dữ liệu đã xác minh để trả lời chính xác yêu cầu này. "
                    "Vui lòng cung cấp thêm chi tiết về sản phẩm hoặc dịch vụ Vingroup."
                )
            else:
                answer = "Xin lỗi, tôi chỉ hỗ trợ thông tin về sản phẩm và dịch vụ thuộc hệ sinh thái Vingroup."
        else:
            answer_parts = []
            catalog_result = observations.get("search_product_catalog")
            if catalog_result is not None:
                if catalog_result and "error" in catalog_result[0]:
                    answer_parts.append(f"Không thể tra cứu danh mục: {catalog_result[0]['error']}")
                elif not catalog_result:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    formatted_products = [
                        f"{product['name']} — {product['price_vnd']:,} VNĐ"
                        for product in catalog_result
                    ]
                    answer_parts.append("Sản phẩm phù hợp: " + "; ".join(formatted_products) + ".")

            ticket_result = observations.get("submit_support_ticket")
            if ticket_result is not None:
                answer_parts.append(
                    f"Yêu cầu hỗ trợ của {ticket_result['customer_name']} đã được ghi nhận "
                    f"với mã {ticket_result['ticket_id']} (ưu tiên {ticket_result['priority']})."
                )
            if needs_customer_name:
                answer_parts.append(
                    "Để ghi nhận yêu cầu hỗ trợ, vui lòng cho tôi biết tên khách hàng."
                )
            answer = " ".join(answer_parts)

        self.trace.append({"step": "final_answer", "content": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
