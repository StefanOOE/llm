#!/usr/bin/env python3
"""Standalone streaming-speed probe for an OpenAI-compatible chat endpoint.

Not part of the container bench harness (run.py / report.py / ab_lmonly.py) --
a quick one-off to eyeball TTFT and tokens/s against a live server:

    python3 common/bench/llm_benchmark.py --url http://localhost:8000/v1 \\
        --key "$API_KEY" --model qwen3-coder-30b
"""
import time
import sys
import json
import argparse
import urllib.request
import urllib.error


def test_speed(api_url, api_key, model, prompt):
    # auto-correct the endpoint if only the base URL was given
    if not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # force streaming so we can time it
        "stream_options": {"include_usage": True}  # exact token count in the last chunk
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    print(f"Sending prompt to {model}...")
    start_time = time.time()
    ttft = None
    first_byte_time = None
    chunk_count = 0
    all_delta_count = 0
    usage = None

    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                if not line:
                    continue

                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    if "[DONE]" in line_str:
                        break

                    try:
                        json_str = line_str[5:].strip()
                        chunk = json.loads(json_str)

                        # the final usage chunk carries an empty choices array
                        if chunk.get("usage"):
                            usage = chunk["usage"]

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # timestamp of the very first delta (reasoning included) = end of prefill
                        has_any_content = delta.get("content") or delta.get("reasoning")
                        if first_byte_time is None and has_any_content:
                            first_byte_time = time.time() - start_time

                        # count every content-bearing chunk (reasoning + visible) for the
                        # bundling-ratio estimate under speculative decoding
                        if has_any_content:
                            all_delta_count += 1

                        # first token with real visible text content
                        if "content" in delta and delta["content"]:
                            if ttft is None:
                                ttft = time.time() - start_time
                                print(f"-> TTFT (first visible token): {ttft:.3f}s\n")

                            # stream the text live
                            sys.stdout.write(delta["content"])
                            sys.stdout.flush()
                            chunk_count += 1  # note: 1 SSE chunk can carry >1 token under speculative decoding
                    except Exception:
                        pass

        end_time = time.time()
        total_time = end_time - start_time
        decode_time = total_time - (first_byte_time if first_byte_time else 0)

        print("\n\n" + "=" * 40)
        print(" METRICS:")
        print("=" * 40)
        if ttft:
            print(f"Time to first token (visible): {ttft:.3f} s")
        else:
            print("Time to first token (visible): N/A")
        print(f"Total request time:            {total_time:.3f} s")
        print(f"Content chunks received:       {chunk_count} (not a 1:1 token mapping under speculative decoding)")

        if usage and usage.get("completion_tokens"):
            completion_tokens = usage["completion_tokens"]
            print(f"Total tokens incl. reasoning (API usage): {completion_tokens}")
            if decode_time > 0:
                speed = completion_tokens / decode_time
                print(f"Response speed (total, exact):            {speed:.2f} tokens/s")
            else:
                print("Response speed (total, exact):            N/A")

            # Visible-text-only estimate: the API gives no separate reasoning
            # token count, so the bundling ratio (real tokens per chunk) from
            # the whole response is applied to the visible content chunks
            # (assumes a similar ratio in the reasoning and content phases).
            if all_delta_count > 0 and ttft is not None:
                bundling_factor = completion_tokens / all_delta_count
                estimated_visible_tokens = chunk_count * bundling_factor
                visible_time = total_time - ttft
                if visible_time > 0:
                    visible_speed = estimated_visible_tokens / visible_time
                    print(f"Response speed (visible text, estimated): {visible_speed:.2f} tokens/s "
                          f"[~{estimated_visible_tokens:.0f} tokens, approx. via bundling ratio]")
        else:
            print("No 'usage' field from the server -- falling back to chunk counting (imprecise under speculative decoding):")
            generation_time = total_time - (ttft if ttft else 0)
            if generation_time > 0 and chunk_count > 0:
                speed = chunk_count / generation_time
                print(f"Response speed (chunk-based):   {speed:.2f} tokens/s")
            else:
                print("Response speed (chunk-based):   N/A")
        print("=" * 40)

    except urllib.error.HTTPError as e:
        print(f"\nHTTP error: {e.code} - {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM API performance tester")
    parser.add_argument("--url", required=True, help="API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--key", required=True, help="API key")
    parser.add_argument("--model", default="qwen3-coder-30b", help="model name")
    parser.add_argument("--prompt", default="Write a detailed explanation of why the sky is blue.", help="test prompt")

    args = parser.parse_args()
    test_speed(args.url, args.key, args.model, args.prompt)
