import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from .agent import build_graph


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(
        description="Run post-vision reading assembler agent with JSON payloads."
    )
    ap.add_argument(
        "--input-json",
        type=str,
        help="Single JSON containing vision_input, router_input, and optional user_preferences.",
    )
    ap.add_argument("--vision-json", type=str, help="Vision output JSON path.")
    ap.add_argument("--router-json", type=str, help="Router output JSON path.")
    ap.add_argument("--prefs-json", type=str, help="Optional user preferences JSON path.")
    ap.add_argument(
        "--payload-json",
        type=str,
        help="Raw JSON string containing vision_input/router_input/user_preferences.",
    )
    args = ap.parse_args()

    payload: dict
    if args.payload_json:
        payload = json.loads(args.payload_json)
    elif args.input_json:
        payload = _load_json(args.input_json)
    else:
        if not args.vision_json or not args.router_json:
            ap.error("Either --input-json / --payload-json OR both --vision-json and --router-json are required.")
        payload = {
            "vision_input": _load_json(args.vision_json),
            "router_input": _load_json(args.router_json),
        }
        if args.prefs_json:
            payload["user_preferences"] = _load_json(args.prefs_json)

    if "vision_input" not in payload or "router_input" not in payload:
        ap.error("Payload must include 'vision_input' and 'router_input'.")

    message = json.dumps(payload, ensure_ascii=False)

    graph = build_graph()
    out = graph.invoke({"messages": [HumanMessage(content=message)]})
    print(out["messages"][-1].content)


if __name__ == "__main__":
    main()
