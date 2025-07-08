import os
os.environ["MKL_THREADING_LAYER"] = "GNU"          # keep MKL happy early

# Set up distributed environment variables
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

# ---- hard-wire a minimal backend so vLLM skips Flash / MLA --------------
from vllm.attention.backends.utils import CommonAttentionState

class _UnitTestBackend:
    """Trivial attention backend that satisfies ModelRunner for unit tests."""
    def __init__(self, *_, **__):
        pass
    @staticmethod
    def get_state_cls():
        return CommonAttentionState
    @staticmethod
    def get_name():
        return "unit_test"
    @staticmethod
    def get_builder_cls():
        class DummyAttnMetadata:
            prefill_metadata = object()
            decode_metadata = object()
            def __init__(self):
                pass
        class DummyModelInput:
            def __init__(self):
                self.attn_metadata = DummyAttnMetadata()
        class DummyBuilder:
            def __init__(self, *a, **k): 
                pass
            def prepare(self, *a, **k): pass
            def build(self, seq_lens=None, query_lens=None, cuda_graph_pad_size=None, batch_size=None, **kwargs):
                return DummyModelInput()
        return DummyBuilder

# Mock the attention backend before importing vLLM modules
import vllm.attention
vllm.attention.get_attn_backend = lambda *a, **kw: _UnitTestBackend(*a, **kw)

# Initialize distributed state for testing
import vllm.distributed.parallel_state as parallel_state
import vllm.distributed as distributed
distributed.init_distributed_environment(world_size=1, rank=0, local_rank=0)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
# ------------------------------------------------------------------------

import queue, torch
from vllm.config import ModelConfig, CacheConfig, VllmConfig, SchedulerConfig, ParallelConfig
from vllm.worker.model_runner import ModelRunner
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
import torch.nn as nn

MODEL_NAME = "facebook/opt-125m" 

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        # Return a tensor with the right shape for hidden states
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    def compute_logits(self, hidden_states, sampling_metadata):
        # Return a tensor with the right shape for logits
        device = hidden_states.device if hidden_states is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.zeros((1, 50257), dtype=torch.float32, device=device)  # Use a reasonable vocab size

class DummySampler:
    def __init__(self):
        pass
    def __call__(self, logits, sampling_metadata):
        # Return a minimal sampler output
        from vllm.model_executor.layers.sampler import SamplerOutput
        from vllm.sequence import CompletionSequenceGroupOutput, SequenceOutput
        # Create a minimal sequence output
        seq_output = SequenceOutput(parent_seq_id=0, output_token=0, logprobs={})
        completion_output = CompletionSequenceGroupOutput(samples=[seq_output], prompt_logprobs=None)
        return SamplerOutput(outputs=[completion_output], sampled_token_ids=torch.tensor([[0]]), sampled_token_probs=None, logprobs=None)

def build_runner():
    mcfg = ModelConfig(
        model                   = MODEL_NAME,
        dtype                   = "float16",
        trust_remote_code       = True,
        task="generate"
    )
    ccfg = CacheConfig(
        block_size=16,
    )
    scfg = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    pcfg = ParallelConfig()
    
    vcfg = VllmConfig(
        model_config = mcfg,
        cache_config = ccfg,
        scheduler_config = scfg,
        parallel_config = pcfg,
        additional_config = {"kv_pruner": "tora.block"},
    )

    runner = ModelRunner(vllm_config=vcfg, is_driver_worker=True)
    runner._result_queue = queue.SimpleQueue()
    runner.model = DummyModel()
    # Create a proper sampler instance
    from vllm.model_executor.layers.sampler import Sampler
    runner.sampler = Sampler()  # No parameters needed
    return runner

def test_tora_scores_decode():
    print("Building runner...")
    
    # Patch the multi-modal input computation at the class level
    from vllm.worker.model_runner import ModelInputForGPUBuilder
    if hasattr(ModelInputForGPUBuilder, "_compute_multi_modal_input"):
        def dummy_compute_multi_modal_input(self, inter_data, seq_group_metadata):
            pass
        ModelInputForGPUBuilder._compute_multi_modal_input = dummy_compute_multi_modal_input
    
    runner = build_runner()
    
    print("Creating sequence data...")

    # PREFILL step: only prompt tokens, is_prompt=True
    prefill_seq_data = SequenceData.from_seqs(
        prompt_token_ids=[1, 2, 3]
    )
    prefill_seq = SequenceGroupMetadata(
        request_id="r0",
        is_prompt=True,
        seq_data={0: prefill_seq_data},
        sampling_params=SamplingParams(temperature=0.0),
        block_tables={0: [0, 1]},
    )

    # DECODE step: prompt + output tokens, is_prompt=False
    decode_seq_data = SequenceData.from_seqs(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[4, 5]
    )
    decode_seq_data.update_num_computed_tokens(5)
    decode_seq = SequenceGroupMetadata(
        request_id="r0",
        is_prompt=False,
        seq_data={0: decode_seq_data},
        sampling_params=SamplingParams(temperature=0.0),
        block_tables={0: [0, 1]},
    )

    print("Running prefill step...")
    mi = runner.prepare_model_input([prefill_seq])
    runner.execute_model(mi, kv_caches=[])
    assert runner._result_queue.empty()
    print("✓ Prefill step completed (no scores as expected)")

    print("Running decode step...")
    mi = runner.prepare_model_input([decode_seq])
    runner.execute_model(mi, kv_caches=[])

    kind, seq_ids, scores = runner._result_queue.get_nowait()
    assert kind == "tora_scores"
    assert len(seq_ids) == len(scores) == 1
    assert scores.dtype == torch.float32
    print(f"✓ Decode step completed - ToRA scores: {scores}")
    print("✓ Test passed!")

if __name__ == "__main__":
    test_tora_scores_decode()
