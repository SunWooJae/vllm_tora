#!/usr/bin/env python3
"""
Token Pruning Example using ToRA Hybrid Scores

This demonstrates how to use the hybrid scoring method to prune
less important tokens during decoding for better efficiency.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class ToRATokenPruner:
    """
    Token pruner using ToRA hybrid scores for efficient decoding.
    """
    
    def __init__(self, pruning_threshold: float = 0.3, min_tokens: int = 4):
        """
        Initialize the token pruner.
        
        Args:
            pruning_threshold: Fraction of tokens to keep (0.3 = keep top 30%)
            min_tokens: Minimum number of tokens to always keep
        """
        self.pruning_threshold = pruning_threshold
        self.min_tokens = min_tokens
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range for fair comparison.
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)  # All equal importance
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def get_token_importance_scores(self, hybrid_scores: Dict[int, Dict]) -> Dict[int, List[float]]:
        """
        Extract unified importance scores for all tokens.
        
        Args:
            hybrid_scores: Output from _compute_hybrid_scores
            
        Returns:
            Dict mapping seq_id to list of token importance scores
        """
        importance_scores = {}
        
        for seq_id, score_data in hybrid_scores.items():
            # Get current token score
            current_score = score_data['current']
            
            # Get cached token scores
            cached_scores = score_data['cached']
            
            # Combine all scores: current token + cached tokens
            all_scores = [current_score] + cached_scores
            
            # Normalize to [0, 1] for fair comparison
            normalized_scores = self.normalize_scores(all_scores)
            
            importance_scores[seq_id] = normalized_scores
        
        return importance_scores
    
    def select_tokens_to_keep(self, importance_scores: List[float], 
                             token_positions: List[int]) -> Tuple[List[int], List[float]]:
        """
        Select which tokens to keep based on importance scores.
        
        Args:
            importance_scores: Normalized importance scores for all tokens
            token_positions: Position indices of tokens
            
        Returns:
            Tuple of (positions_to_keep, scores_of_kept_tokens)
        """
        if len(importance_scores) <= self.min_tokens:
            return token_positions, importance_scores
        
        # Sort tokens by importance (descending)
        token_score_pairs = list(zip(token_positions, importance_scores))
        token_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many tokens to keep
        num_tokens = len(importance_scores)
        num_to_keep = max(self.min_tokens, int(num_tokens * self.pruning_threshold))
        
        # Select top tokens
        kept_pairs = token_score_pairs[:num_to_keep]
        kept_positions = [pos for pos, _ in kept_pairs]
        kept_scores = [score for _, score in kept_pairs]
        
        return kept_positions, kept_scores
    
    def prune_kv_cache(self, kv_cache_scores: Dict[int, List[float]], 
                      hybrid_scores: Dict[int, Dict]) -> Dict[int, List[int]]:
        """
        Determine which KV cache tokens to keep for each sequence.
        
        Args:
            kv_cache_scores: Raw KV cache scores
            hybrid_scores: Hybrid scoring results
            
        Returns:
            Dict mapping seq_id to list of token positions to keep
        """
        pruning_decisions = {}
        
        for seq_id in hybrid_scores.keys():
            # Get importance scores for this sequence
            importance_scores = self.get_token_importance_scores({seq_id: hybrid_scores[seq_id]})
            seq_importance = importance_scores[seq_id]
            
            # Create position indices (0 = current token, 1+ = cached tokens)
            token_positions = list(range(len(seq_importance)))
            
            # Select tokens to keep
            kept_positions, kept_scores = self.select_tokens_to_keep(
                seq_importance, token_positions)
            
            # Filter out current token (position 0) from KV cache decisions
            kv_cache_positions = [pos - 1 for pos in kept_positions if pos > 0]
            
            pruning_decisions[seq_id] = kv_cache_positions
        
        return pruning_decisions

def demonstrate_pruning():
    """
    Demonstrate how token pruning works with example hybrid scores.
    """
    
    # Example hybrid scores from the logs
    example_hybrid_scores = {
        0: {
            'current': 199.404541015625,
            'cached': [71.0095443725586, 85.47956848144531, 122.72186279296875],
            'hybrid': 119.79964309692383,
            'weights': {
                'current_weight': 0.38,
                'cached_weight': 0.62,
                'recency_weights': [1.0, 0.9, 0.81],
                'importance_weights': [0.2, 0.3, 0.5],
                'combined_weights': [0.2, 0.27, 0.405]
            },
            'num_cached_tokens': 3
        }
    }
    
    # Initialize pruner
    pruner = ToRATokenPruner(pruning_threshold=0.6, min_tokens=2)
    
    # Get importance scores
    importance_scores = pruner.get_token_importance_scores(example_hybrid_scores)
    
    print("=== ToRA Token Pruning Demonstration ===")
    print()
    
    for seq_id, scores in importance_scores.items():
        print(f"Sequence {seq_id}:")
        print(f"  Raw scores: {scores}")
        print(f"  Token count: {len(scores)}")
        
        # Simulate pruning decision
        token_positions = list(range(len(scores)))
        kept_positions, kept_scores = pruner.select_tokens_to_keep(scores, token_positions)
        
        print(f"  Tokens to keep: {kept_positions}")
        print(f"  Kept scores: {[f'{s:.3f}' for s in kept_scores]}")
        print(f"  Pruning ratio: {len(kept_positions)}/{len(scores)} = {len(kept_positions)/len(scores):.1%}")
        print()
        
        # Show KV cache pruning
        kv_decisions = pruner.prune_kv_cache({}, example_hybrid_scores)
        print(f"  KV cache positions to keep: {kv_decisions[seq_id]}")
        print()

def explain_pruning_benefits():
    """
    Explain the benefits of token pruning for decoding efficiency.
    """
    
    print("=== Benefits of ToRA Token Pruning ===")
    print()
    
    benefits = [
        ("Memory Efficiency", 
         "Reduce KV cache memory usage by keeping only important tokens"),
        ("Computation Speed", 
         "Fewer tokens = faster attention computation and decoding"),
        ("Quality Preservation", 
         "Keep most important tokens to maintain generation quality"),
        ("Adaptive Pruning", 
         "Pruning threshold can be adjusted based on model and task"),
        ("Real-time Decision", 
         "Make pruning decisions during each decode step")
    ]
    
    for benefit, description in benefits:
        print(f"✓ {benefit}: {description}")
    
    print()
    print("=== Pruning Strategy Considerations ===")
    print()
    
    strategies = [
        ("Conservative (threshold=0.8)", "Keep 80% of tokens, minimal quality loss"),
        ("Balanced (threshold=0.6)", "Keep 60% of tokens, good efficiency/quality trade-off"),
        ("Aggressive (threshold=0.4)", "Keep 40% of tokens, maximum efficiency"),
        ("Adaptive", "Adjust threshold based on sequence length and model confidence")
    ]
    
    for strategy, description in strategies:
        print(f"• {strategy}: {description}")

if __name__ == "__main__":
    demonstrate_pruning()
    explain_pruning_benefits()
    
    print("=== Integration with vLLM ===")
    print()
    print("To integrate this with vLLM decoding:")
    print("1. Extract hybrid scores from model_runner.py")
    print("2. Apply pruning decisions to KV cache")
    print("3. Update attention metadata to skip pruned tokens")
    print("4. Monitor quality vs efficiency trade-offs") 