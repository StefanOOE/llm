"""vLLM 0.24.0 (NGC 26.07): ModelOpt get_quant_method() gives an EXCLUDED
ParallelLMHead UnquantizedLinearMethod(), which has no tie_weights() ->
Gemma 4 (tie_word_embeddings, lm_head excluded) crashes at load with
NotImplementedError in base_config.tie_weights. Return
UnquantizedEmbeddingMethod() (has tie_weights) for the LM head instead;
plain linear layers keep UnquantizedLinearMethod. Newer vLLM does this.
"""
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py")
s = p.read_text()
old = ("            if isinstance(layer, (LinearBase, ParallelLMHead)):\n"
       "                return UnquantizedLinearMethod()\n"
       "            return None")
new = ("            if isinstance(layer, ParallelLMHead):\n"
       "                return UnquantizedEmbeddingMethod()  # has tie_weights (tied lm_head)\n"
       "            if isinstance(layer, LinearBase):\n"
       "                return UnquantizedLinearMethod()\n"
       "            return None")
# the import is a one-liner in vLLM 0.24 (NGC 26.07) and a multi-line block
# in 0.33 (NGC 26.11); handle both
imp_old = "from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead\n"
imp_new = ("from vllm.model_executor.layers.vocab_parallel_embedding import (\n"
           "    ParallelLMHead,\n    UnquantizedEmbeddingMethod,\n)\n")
blk_old = ("from vllm.model_executor.layers.vocab_parallel_embedding import (\n"
           "    ParallelLMHead,\n")
blk_new = blk_old + "    UnquantizedEmbeddingMethod,\n"
assert s.count(old) == 2, s.count(old)
if s.count(imp_old) == 1:
    s = s.replace(imp_old, imp_new)
else:
    assert s.count(blk_old) == 1, "ParallelLMHead import not found"
    s = s.replace(blk_old, blk_new)
p.write_text(s.replace(old, new))
print("patched modelopt.py: excluded ParallelLMHead -> UnquantizedEmbeddingMethod (2 sites)")
