# quick interactive helper
import inspect, textwrap
from vllm.worker.model_runner import ModelRunner

print(textwrap.dedent(inspect.getsource(ModelRunner.__init__)))
