#!/usr/bin/env python3
import time
import sys
import json
import argparse
import urllib.request
import urllib.error

def test_speed(api_url, api_key, model, prompt):
    # Endpunkt automatisch korrigieren, falls unvollständig
    if not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # Streaming erzwingen für zeitliche Messung
        "stream_options": {"include_usage": True}  # liefert exakte Tokenzahl im letzten Chunk
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    print(f"Sende Prompt an {model}...")
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

                        # Der finale Usage-Chunk hat ein leeres choices-Array
                        if chunk.get("usage"):
                            usage = chunk["usage"]

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # Zeitpunkt des allerersten Deltas (auch Reasoning) = Ende Prefill
                        has_any_content = delta.get("content") or delta.get("reasoning")
                        if first_byte_time is None and has_any_content:
                            first_byte_time = time.time() - start_time

                        # Zählt alle Content-tragenden Chunks (Reasoning + sichtbar), für die
                        # Schätzung des Bündelungsverhältnisses bei Speculative Decoding
                        if has_any_content:
                            all_delta_count += 1

                        # Erkennt den ersten Token mit echtem sichtbarem Textinhalt
                        if "content" in delta and delta["content"]:
                            if ttft is None:
                                ttft = time.time() - start_time
                                print(f"-> TTFT (Erster sichtbarer Token nach): {ttft:.3f}s\n")

                            # Gibt den Text live aus
                            sys.stdout.write(delta["content"])
                            sys.stdout.flush()
                            chunk_count += 1  # Achtung: 1 SSE-Chunk kann bei Speculative Decoding >1 Token enthalten
                    except Exception:
                        pass

        end_time = time.time()
        total_time = end_time - start_time
        decode_time = total_time - (first_byte_time if first_byte_time else 0)

        print("\n\n" + "="*40)
        print(" METRIKEN:")
        print("="*40)
        if ttft:
            print(f"Time to First Token (sichtbar): {ttft:.3f} Sekunden")
        else:
            print("Time to First Token (sichtbar): N/A")
        print(f"Gesamtzeit der Anfrage:          {total_time:.3f} Sekunden")
        print(f"Content-Chunks empfangen:        {chunk_count} (kein 1:1-Tokenmapping bei Speculative Decoding)")

        if usage and usage.get("completion_tokens"):
            completion_tokens = usage["completion_tokens"]
            print(f"Tokens gesamt inkl. Reasoning (API usage): {completion_tokens}")
            if decode_time > 0:
                speed = completion_tokens / decode_time
                print(f"Antwortgeschwindigkeit (gesamt, exakt):    {speed:.2f} Tokens/Sekunde")
            else:
                print("Antwortgeschwindigkeit (gesamt, exakt):    N/A")

            # Schätzung nur für sichtbaren Text: die API liefert keine getrennte
            # Reasoning-Tokenzahl, daher wird das Bündelungsverhältnis (echte
            # Tokens pro Chunk) aus der Gesamtantwort auf die sichtbaren
            # Content-Chunks übertragen (Annahme: ähnliches Verhältnis in
            # Reasoning- und Content-Phase).
            if all_delta_count > 0 and ttft is not None:
                bundling_factor = completion_tokens / all_delta_count
                estimated_visible_tokens = chunk_count * bundling_factor
                visible_time = total_time - ttft
                if visible_time > 0:
                    visible_speed = estimated_visible_tokens / visible_time
                    print(f"Antwortgeschwindigkeit (sichtbarer Text, geschätzt): {visible_speed:.2f} Tokens/Sekunde "
                          f"[~{estimated_visible_tokens:.0f} Tokens, Näherung über Bündelungsverhältnis]")
        else:
            print("Kein 'usage'-Feld vom Server erhalten — Fallback auf Chunk-Zählung (ungenau bei Speculative Decoding):")
            generation_time = total_time - (ttft if ttft else 0)
            if generation_time > 0 and chunk_count > 0:
                speed = chunk_count / generation_time
                print(f"Antwortgeschwindigkeit (Chunk-basiert):     {speed:.2f} Tokens/Sekunde")
            else:
                print("Antwortgeschwindigkeit (Chunk-basiert):     N/A")
        print("="*40)
        
    except urllib.error.HTTPError as e:
        print(f"\nHTTP Fehler: {e.code} - {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"\nFehler: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM API Performance Tester")
    parser.add_argument("--url", required=True, help="API Base URL (z.B. https://server.de)")
    parser.add_argument("--key", required=True, help="API Key")
    parser.add_argument("--model", default="qwen3.8-27b-uncensored", help="Modellname")
    parser.add_argument("--prompt", default="Schreibe eine ausführliche Erklärung, warum der Himmel blau ist.", help="Test-Prompt")
    
    args = parser.parse_args()
    test_speed(args.url, args.key, args.model, args.prompt)

