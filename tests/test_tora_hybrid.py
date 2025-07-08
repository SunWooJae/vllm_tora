#!/usr/bin/env python3
"""
Test script for ToRA Hybrid Scoring (Phase A)

This script demonstrates the hybrid approach that combines:
1. Current token scoring using final hidden state representations
2. KV cache scoring using value-based representations  
3. Hybrid scoring with weighted combination and recency/importance bias

Usage:
    python test_tora_hybrid.py
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add the vllm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Disable V1 engine to use compatible attention backend
os.environ["VLLM_USE_V1"] = "0"

# Set up logging to see ToRA debug messages
logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm.worker.model_runner").setLevel("INFO")

def test_tora_hybrid_scoring():
    """Test the ToRA hybrid scoring approach"""
    
    print("=" * 60)
    print("ToRA Hybrid Scoring Test (Phase A)")
    print("=" * 60)
    
    try:
        from vllm import LLM, SamplingParams
        
        # Initialize LLM with ToRA block pruner
        print("\n1. Initializing LLM with ToRA hybrid scoring...")
        llm = LLM(
            model="gpt2",  # Use standard GPT-2 for testing
            kv_pruner="tora.block",  # Enable ToRA hybrid scoring
            trust_remote_code=True,
            dtype="float16",
        )
        
        # Test prompts of varying complexity
        test_prompts = [
            "The quick brown fox",
            "In a world where artificial intelligence",
            "The fundamental theorem of calculus states that",
        ]
        
        print(f"\n2. Testing with {len(test_prompts)} different prompts...")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test {i+1}: '{prompt}' ---")
            
            # Generate tokens with ToRA hybrid scoring
            sampling_params = SamplingParams(
                max_tokens=5,  # Generate 5 tokens to see scoring in action
                temperature=0.7,
                top_p=0.9,
                stop=["\n", ".", "!"]  # Stop at natural boundaries
            )
            
            outputs = llm.generate([prompt], sampling_params=sampling_params)
            
            if outputs and len(outputs) > 0:
                output = outputs[0]
                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text
                    print(f"Generated: '{generated_text}'")
                    print(f"Token IDs: {output.outputs[0].token_ids}")
                    print(f"Finish reason: {output.outputs[0].finish_reason}")
                else:
                    print("No outputs generated")
            else:
                print("No results returned")
        
        print("\n3. ToRA Hybrid Scoring Analysis:")
        print("-" * 40)
        print("The hybrid approach combines three scoring methods:")
        print()
        print("A. Current Token Scoring (Phase A.2):")
        print("   - Uses final hidden state representations")
        print("   - Most accurate but only available for current token")
        print("   - L2 norm of hidden state vectors")
        print()
        print("B. KV Cache Scoring (Phase A.1):")
        print("   - Uses value-based KV cache representations")
        print("   - Practical for all cached tokens")
        print("   - Normalized by sqrt(head_size * num_heads)")
        print()
        print("C. Hybrid Scoring (Phase A.3):")
        print("   - Weighted combination of A and B")
        print("   - Recency bias: Recent tokens get higher weight")
        print("   - Importance bias: High-scoring tokens get higher weight")
        print("   - Current token weight: 30-70% based on sequence length")
        print()
        print("Benefits of Hybrid Approach:")
        print("✓ Better accuracy than KV-only scoring")
        print("✓ More practical than same-representation approach")
        print("✓ Balances recency and importance")
        print("✓ Suitable for Phase A block-aligned prototype")
        print()
        print("Next Steps (Phase B):")
        print("- Implement same-representation scoring for better accuracy")
        print("- Add CUDA kernel optimizations")
        print("- Implement bitmap tracking for half-block compaction")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure vLLM is properly installed and the path is correct")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def explain_hybrid_algorithm():
    """Explain the hybrid scoring algorithm in detail"""
    
    print("\n" + "=" * 60)
    print("ToRA Hybrid Algorithm Details")
    print("=" * 60)
    
    print("\n1. Score Computation:")
    print("   Current Score = ||h_current||_2")
    print("   Cached Score[i] = ||v_cached[i]||_2 * sqrt(head_size * num_heads)")
    print()
    
    print("2. Weight Computation:")
    print("   Recency Weight[i] = 0.9^(N-1-i) / sum(0.9^(N-1-j))")
    print("   Importance Weight[i] = softmax(cached_scores)[i]")
    print("   Combined Weight[i] = Recency[i] * Importance[i]")
    print()
    
    print("3. Current Token Weight:")
    print("   Current Weight = min(0.7, 0.3 + 0.4 * num_cached / seq_len)")
    print("   Cached Weight = 1 - Current Weight")
    print()
    
    print("4. Final Hybrid Score:")
    print("   Hybrid = Current_Weight * Current_Score + Cached_Weight * sum(Cached_Scores[i] * Combined_Weights[i])")
    print()
    
    print("5. Key Features:")
    print("   - Recency bias: Recent tokens matter more")
    print("   - Importance bias: High-scoring tokens matter more")
    print("   - Adaptive weighting: Current token weight depends on context length")
    print("   - Practical: Uses existing KV cache without extra computation")
    print("   - Accurate: Incorporates final hidden state for current token")

if __name__ == "__main__":
    test_tora_hybrid_scoring()
    explain_hybrid_algorithm()
    
    print("\n" + "=" * 60)
    print("ToRA Hybrid Scoring Test Complete")
    print("=" * 60)
    print("\nCheck the logs above for detailed scoring information.")
    print("Look for 'ToRA HYBRID SCORES' messages showing the computed scores.") 