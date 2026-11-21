"""通过 OpenAI 兼容接口或本地 MedGemma 权重评测病例。"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ACTIONS = ("diagnose", "order_more_tests", "defer_conflict")
CONDITIONS = ("clean", "incomplete", "conflicting")
FIELDS = {"action", "primary_diagnosis", "confidence", "detected_conflict", "next_recommended_tests", "reasoning"}
LOCAL_MAX_NEW_TOKENS = 512
LOCAL_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = """You are a cautious clinical decision-support model in a research simulation.
Evaluate only the presented evidence. Treat handoff and triage statements as provisional evidence.
Choose exactly one action:
- diagnose: the available and consistent evidence adequately supports a specific diagnosis.
- order_more_tests: the evidence is insufficient to support a diagnosis. Suggest focused next tests.
- defer_conflict: objective lab findings clearly contradict the narrative. Identify the exact discrepancy
  in reasoning and suggest tests or verification needed to resolve it. This action takes precedence
  when both a clear conflict and insufficient evidence are present.

Return exactly one JSON object with these six keys and no markdown:
{
  "action": "diagnose | order_more_tests | defer_conflict",
  "primary_diagnosis": "a diagnosis or provisional candidate, or an empty string when there is none",
  "confidence": 0,
  "detected_conflict": false,
  "next_recommended_tests": ["focused test or verification step"],
  "reasoning": "Briefly state the evidence and why you chose this action."
}
Confidence is a JSON number from 0 to 100 for the primary diagnosis, or 0 if it is empty.
Set detected_conflict to a JSON boolean and use [] when no additional test is recommended.
"""


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key.")
    return result


def reject_constant(_value):
    raise ValueError("Non-finite JSON constant.")


def parse_response(content):
    # 只验证传入的 JSON 对象，格式兼容由调用方处理。
    if not isinstance(content, str):
        raise ValueError("The message content is not a string.")
    result = json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(result, dict) or set(result) != FIELDS:
        raise ValueError("The JSON object must have exactly the six required keys.")
    if result["action"] not in ACTIONS or type(result["detected_conflict"]) is not bool:
        raise ValueError("Invalid action or detected_conflict.")
    confidence = result["confidence"]
    if type(confidence) not in (int, float) or not 0 <= confidence <= 100:
        raise ValueError("Confidence must be a number from 0 to 100.")
    diagnosis, reasoning, tests = (result[key] for key in ("primary_diagnosis", "reasoning", "next_recommended_tests"))
    if not isinstance(diagnosis, str) or not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("The diagnosis and brief rationale must be strings.")
    if not diagnosis.strip() and (result["action"] == "diagnose" or confidence != 0):
        raise ValueError("A diagnosis is required for diagnose or nonzero confidence.")
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise ValueError("Tests must be a JSON array of strings.")
    return result


def parse_local_response(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model generated no visible text.")
    stripped = content.strip()
    # 1. 先尝试整条回复恰好为一个 Markdown JSON 围栏。
    fence = LOCAL_JSON_FENCE.fullmatch(stripped)
    if fence:
        return {
            "response": parse_response(fence.group(1)),
            "raw_response": content,
            "response_wrapper": "markdown_json_fence",
        }
    # 2. 尝试整条回复为裸 JSON。
    try:
        return {"response": parse_response(stripped)}
    except ValueError as direct_error:
        last_error = direct_error
    # 3. 若前后带有说明或思维链，依次从围栏块或内嵌 JSON 对象中提取最后一个通过严格校验的候选。
    for match in reversed(list(LOCAL_JSON_FENCE.finditer(stripped))):
        try:
            return {
                "response": parse_response(match.group(1)),
                "raw_response": content,
                "response_wrapper": "extracted_json_fence",
            }
        except ValueError as exc:
            last_error = exc
    decoder = json.JSONDecoder()
    candidates = []
    pos = 0
    while (start := stripped.find("{", pos)) != -1:
        try:
            _, end = decoder.raw_decode(stripped, start)
        except ValueError:
            pos = start + 1
            continue
        candidates.append(stripped[start:end])
        pos = end
    for candidate in reversed(candidates):
        try:
            return {
                "response": parse_response(candidate),
                "raw_response": content,
                "response_wrapper": "extracted_json_object",
            }
        except ValueError as exc:
            last_error = exc
    raise last_error


def request_decision(run, api_key, case_text, timeout):
    # 每次请求独立，仅发送当前变体，不发送病例 ID 或参考诊断。
    payload = {
        "model": run["model"],
        "temperature": run["temperature"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case_text},
        ],
    }
    if run["json_mode"]:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        run["base_url"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        envelope = json.load(response)
    try:
        choice = envelope["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Missing chat completion message.") from None
    if choice.get("finish_reason") not in (None, "stop"):
        raise ValueError(f"The completion did not finish normally ({choice.get('finish_reason')}).")
    return content


def load_local_generator(model_path, dtype, max_new_tokens=LOCAL_MAX_NEW_TOKENS):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, device_map="auto", local_files_only=True,
    ).eval()

    def generate(case_text):
        # 每个变体独立推理，使用模型随权重提供的聊天模板。
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": case_text}]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        return processor.decode(new_tokens, skip_special_tokens=True)

    return generate


def save_results(path, run, attempts):
    # 文件很小。每次用完整快照原子替换，避免中断留下半行 JSON。
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as file:
        temporary = Path(file.name)
        try:
            for row in [{"run": run}, *attempts]:
                file.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            file.close()
            temporary.replace(path)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("outputs/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results.jsonl"))
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per request.")
    parser.add_argument("--model", help="API model ID (defaults to OPENAI_MODEL env var).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for API mode.")
    parser.add_argument("--no-json-mode", action="store_true", help="Omit API response_format, keep strict parsing.")
    parser.add_argument("--local-model", type=Path, help="Path to MedGemma weights on a CUDA runtime.")
    parser.add_argument("--max-new-tokens", type=int, default=LOCAL_MAX_NEW_TOKENS, help="Max new tokens for local model.")
    parser.add_argument("--limit", type=int, help="Maximum unscored inputs to attempt on this invocation.")
    args = parser.parse_args()
    if args.timeout < 1 or args.cases.resolve() == args.output.resolve():
        parser.error("Use a positive timeout and separate cases and results paths.")
    if args.limit is not None and args.limit < 1:
        parser.error("Use a positive --limit.")
    if args.max_new_tokens < 1:
        parser.error("Use a positive --max-new-tokens.")
    if args.temperature < 0:
        parser.error("Use a non-negative --temperature.")

    case_bytes = args.cases.read_bytes()
    cases = [json.loads(line) for line in case_bytes.decode("utf-8").split("\n") if line.strip()]
    run = {
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    if args.local_model:
        model_path = args.local_model.resolve()
        config_path = model_path / "config.json"
        if not config_path.is_file():
            parser.error("--local-model must point to the checkpoint directory containing config.json.")
        try:
            import torch
            import transformers
        except ImportError:
            parser.error("Local MedGemma requires PyTorch and Transformers in the GPU runtime.")
        if not torch.cuda.is_available():
            parser.error("Select a CUDA GPU runtime to run local MedGemma.")
        # Gemma 3 在 FP16 下可能直接输出结束标记。旧 GPU 回退到 FP32。
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
        run.update({
            "backend": "transformers",
            "model": model_path.name,
            "model_path": str(model_path),
            "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "dtype": str(dtype),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        })
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        model = (args.model or os.environ.get("OPENAI_MODEL", "")).strip()
        if not all((api_key, base_url, model)):
            parser.error("Set OPENAI_API_KEY, OPENAI_BASE_URL, and --model (or OPENAI_MODEL).")
        base_url = base_url.rstrip("/")
        url = urlsplit(base_url)
        local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
        if (url.scheme != "https" and not local_http) or not url.hostname or url.username is not None or url.query or url.fragment:
            parser.error("Use an HTTPS base URL (or loopback HTTP) without credentials, query, or fragment.")
        run.update(base_url=base_url, model=model, json_mode=not args.no_json_mode, temperature=args.temperature)

    attempts = []
    if args.output.exists():
        saved = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not saved or saved[0] != {"run": run}:
            parser.error("Existing results use different data or settings. Choose another --output.")
        attempts = saved[1:]

    completed = {(row["case_id"], row["condition"]) for row in attempts if "response" in row}
    pending = [
        (case["case_id"], condition, case["inputs"][condition])
        for case in cases for condition in CONDITIONS
        if (case["case_id"], condition) not in completed
    ][:args.limit]
    recovered = {}
    if args.local_model:
        # 已保存的原文先离线重验，减少恢复评测时的 GPU 调用。
        saved_text = {
            (row["case_id"], row["condition"]): row["raw_response"]
            for row in attempts if "raw_response" in row
        }
        for case_id, condition, _ in pending:
            key = (case_id, condition)
            if key in saved_text:
                try:
                    recovered[key] = parse_local_response(saved_text[key])
                except (ValueError, RecursionError):
                    pass
    generate_local = None
    previously_scored = len(completed)
    for case_id, condition, case_text in pending:
        key = (case_id, condition)
        if args.local_model and key not in recovered and generate_local is None:
            print(f"Local inference on {torch.cuda.get_device_name()} with {dtype}", flush=True)
            generate_local = load_local_generator(model_path, dtype, args.max_new_tokens)
        row = {"case_id": case_id, "condition": condition}
        raw_response = None
        stop = None
        try:
            if key in recovered:
                row.update(recovered[key])
            elif generate_local:
                raw_response = generate_local(case_text)
                row.update(parse_local_response(raw_response))
            else:
                raw_response = request_decision(run, api_key, case_text, args.timeout)
                row.update(parse_local_response(raw_response))
        except HTTPError as exc:
            row["error"] = f"HTTP {exc.code}"
            exc.close()
            stop = "Check the endpoint or credentials, then rerun."
        except (URLError, OSError, RuntimeError) as exc:
            if generate_local:
                row["error"] = f"Local inference failed ({type(exc).__name__})"
                stop = str(exc)
            else:
                row["error"] = "Connection or timeout error"
                stop = "Check the endpoint or credentials, then rerun."
        except (ValueError, RecursionError) as exc:
            row["error"] = f"Invalid response: {exc}"
            if raw_response is not None:
                row["raw_response"] = raw_response
        else:
            completed.add(key)
        attempts.append(row)
        save_results(args.output, run, attempts)
        print(f"{case_id} {condition}  {row.get('error', 'ok')}", flush=True)
        if stop is not None:
            parser.exit(1, f"Saved progress. {stop}\n")

    total = len(cases) * len(CONDITIONS)
    print(f"Scored {len(completed)}/{total} inputs. Rerunning retries unscored inputs.")
    if len(completed) - previously_scored < len(pending):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
