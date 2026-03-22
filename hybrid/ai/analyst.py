"""Claude API integration — the AI analyst.

Sends the system prompt + tools to Claude, processes tool calls,
validates orders through the risk layer, and returns the analysis.
"""

import json
import logging
from typing import Any

import anthropic

from hybrid.ai.prompts import build_system_prompt
from hybrid.broker.tools import TOOLS, execute_tool
from hybrid.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from hybrid.risk.validator import (
    get_daily_state,
    should_force_close_all,
    validate_close,
    validate_new_order,
)
from hybrid.broker import alpaca

logger = logging.getLogger(__name__)

# Max tool-use rounds per cycle to prevent runaway loops
MAX_TOOL_ROUNDS = 15


def run_analysis_cycle() -> dict:
    """Run one complete analysis cycle.

    1. Build system prompt with current state
    2. Send to Claude with tools
    3. Process tool calls in a loop
    4. Validate any orders through risk layer
    5. Return structured result

    Returns:
        dict with keys: action, reasoning, trades, errors, token_usage
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    daily_state = get_daily_state()

    # Check if we need to force close everything
    force_close = should_force_close_all()
    user_message = "Run your analysis cycle now."
    if force_close:
        user_message = (
            "URGENT: It is past hard close time. Close ALL open positions immediately "
            "using market orders. Do not enter any new trades."
        )

    system_prompt = build_system_prompt(daily_state)

    messages = [{"role": "user", "content": user_message}]

    total_input_tokens = 0
    total_output_tokens = 0
    trades_executed = []
    errors = []
    final_text = ""

    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error("Claude API error: %s", e)
            errors.append(f"API error: {e}")
            break

        # Track tokens
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Process response content blocks
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                final_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(block)

        # If no tool calls, Claude is done
        if not tool_calls:
            break

        # Process each tool call
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.input
            tool_id = tool_call.id

            logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_args, default=str))

            # Intercept order placement for validation
            if tool_name == "place_order":
                result = _validated_place_order(tool_args, trades_executed, errors)
            elif tool_name == "close_position":
                result = _validated_close_position(tool_args, trades_executed, errors)
            else:
                result = execute_tool(tool_name, tool_args)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result,
            })

        # Add assistant message and tool results for next round
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # If Claude signaled stop, break
        if response.stop_reason == "end_turn":
            break

    # Parse the action from Claude's final text
    action = _parse_action(final_text)

    result = {
        "action": action,
        "reasoning": final_text,
        "trades": trades_executed,
        "errors": errors,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost": round(
                total_input_tokens / 1_000_000 * 3 +
                total_output_tokens / 1_000_000 * 15, 4
            ),
        },
        "daily_state": get_daily_state(),
    }

    logger.info(
        "Cycle complete: action=%s trades=%d errors=%d tokens=%d+%d cost=$%.4f",
        action, len(trades_executed), len(errors),
        total_input_tokens, total_output_tokens,
        result["token_usage"]["estimated_cost"],
    )

    return result


def _validated_place_order(args: dict, trades: list, errors: list) -> str:
    """Validate an order through the risk layer before execution."""
    try:
        # Get current state for validation
        account = alpaca.get_account()
        positions = alpaca.get_positions()

        validation = validate_new_order(
            symbol=args["symbol"],
            qty=args["qty"],
            side=args["side"],
            order_type=args["order_type"],
            limit_price=args.get("limit_price"),
            current_positions=positions,
            account=account,
        )

        if not validation["approved"]:
            errors.append(f"Order blocked: {validation['reason']}")
            return json.dumps({
                "error": "ORDER BLOCKED BY RISK VALIDATOR",
                "reason": validation["reason"],
                "violations": validation.get("violations", []),
            })

        # Validation passed — execute
        result = execute_tool("place_order", args)
        result_data = json.loads(result)

        if "error" not in result_data:
            trades.append({
                "type": "entry",
                "symbol": args["symbol"],
                "side": args["side"],
                "qty": args["qty"],
                "order_type": args["order_type"],
                "limit_price": args.get("limit_price"),
                "order_id": result_data.get("order_id"),
                "status": result_data.get("status"),
            })

        return result

    except Exception as e:
        error_msg = f"Order validation failed: {e}"
        errors.append(error_msg)
        return json.dumps({"error": error_msg})


def _validated_close_position(args: dict, trades: list, errors: list) -> str:
    """Validate a position close before execution."""
    try:
        positions = alpaca.get_positions()
        validation = validate_close(args["symbol"], positions)

        if not validation["approved"]:
            errors.append(f"Close blocked: {validation['reason']}")
            return json.dumps({
                "error": "CLOSE BLOCKED",
                "reason": validation["reason"],
            })

        result = execute_tool("close_position", args)
        result_data = json.loads(result)

        if "error" not in result_data:
            trades.append({
                "type": "exit",
                "symbol": args["symbol"],
                "qty": args.get("qty"),
                "order_id": result_data.get("order_id"),
            })

        return result

    except Exception as e:
        error_msg = f"Close validation failed: {e}"
        errors.append(error_msg)
        return json.dumps({"error": error_msg})


def _parse_action(text: str) -> str:
    """Extract the action from Claude's summary text."""
    text_upper = text.upper()
    if "NO_TRADE" in text_upper or "NO TRADE" in text_upper:
        return "NO_TRADE"
    if "ENTERED" in text_upper:
        return "ENTERED"
    if "EXITED" in text_upper:
        return "EXITED"
    if "HOLD" in text_upper:
        return "HOLD"
    if "FORCE CLOSE" in text_upper or "FORCE_CLOSE" in text_upper:
        return "FORCE_CLOSE"
    return "ANALYZED"
