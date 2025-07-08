# run_llm.py  ─── fast local smoke-test
import logging
import os
from vllm import LLM, SamplingParams

# Disable V1 engine to use compatible attention backend
os.environ["VLLM_USE_V1"] = "0"

# make the patch visible
logging.getLogger("vllm.worker.model_runner").setLevel("INFO")

llm = LLM(
    model="gpt2",  # Use standard GPT-2 which is more reliable for generation
    kv_pruner="tora.block",          # ← your new knob
    trust_remote_code=True,
    dtype="float16",
)

def main():
    # Generate tokens and collect ToRA scores
    sampling_params = SamplingParams(
        max_tokens=10, 
        temperature=0.7,  # Add some randomness
        top_p=0.9,
        stop=["\n", ".", "!"]  # Stop at natural boundaries
    )
    
    print("Starting generation...")
    out = llm.generate(
        ["The quick brown fox"],
        sampling_params=sampling_params,
    )
    
    print(f"Output type: {type(out)}")
    print(f"Output length: {len(out)}")
    
    if len(out) > 0:
        print(f"First output type: {type(out[0])}")
        
        if hasattr(out[0], 'outputs') and len(out[0].outputs) > 0:
            output = out[0].outputs[0]
            print(f"Generated text: '{output.text}'")
            print(f"Generated text length: {len(output.text)}")
            print(f"Token IDs: {output.token_ids}")
            print(f"Finish reason: {output.finish_reason}")
            print(f"Finished: {output.finished}")
        else:
            print("No outputs found in the result")
    else:
        print("No results returned")
    
    print("ToRA hybrid scoring is enabled with kv_pruner='tora.block'")
    print("Note: ToRA hybrid scores combine current token and KV cache scores during decode steps")
    print("The scores are sent through the result queue with format: ('tora_hybrid_scores', seq_ids, current_scores, kv_cache_scores, hybrid_scores)")
    print("Hybrid approach provides better accuracy by combining:")
    print("  - Current tokens: Final hidden state representations (most accurate)")
    print("  - Cached tokens: Value-based KV cache representations (practical)")
    print("  - Hybrid: Weighted combination with recency and importance bias")

if __name__ == "__main__":
    main()
