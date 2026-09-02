# Corrected tokenizer config

`tokenizer_config.json` here is `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8`'s
own `tokenizer_config.json` with one change:

```
"tokenizer_class": "LlamaTokenizer"   ->   "PreTrainedTokenizerFast"
```

(and a few now-irrelevant SentencePiece-only keys dropped: `legacy`,
`sp_model_kwargs`, `use_default_system_prompt`, `add_prefix_space`).

Why: `transformers` loads `LlamaTokenizer`/`LlamaTokenizerFast` with a
SentencePiece "metaspace" decoder (`▁` → space). DeepSeek-Coder-V2's vocab is
GPT-2 byte-level BPE (`Ġ` / `Ċ`), so that decoder drops every space and
newline from generated text. Forcing the generic `PreTrainedTokenizerFast`
keeps the `ByteLevel` decoder that `tokenizer.json` already declares.

`tokenizer.json` itself is **not** vendored here — `../model.env` copies it
verbatim from the model's downloaded snapshot at start time and drops this
config next to it under `$HF_CACHE/tokenizer-fixes/`, then serves that dir via
`--tokenizer`. The model weights and tokenizer are under the DeepSeek license
(see the model card); this repo only carries the one-line config fix.
