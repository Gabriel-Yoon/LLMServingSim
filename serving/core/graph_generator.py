import os
import sys
from .request import *
from .utils import _TMP_DIR
from .logger import get_logger

logger = get_logger("GraphGenerator")

def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None):

    cwd = os.getcwd()  # always astra-sim/ (set at startup by __main__.py)

    # Add chakra to sys.path once so the import resolves without chdir.
    chakra_dir = os.path.join(cwd, "extern/graph_frontend/chakra")
    if chakra_dir not in sys.path:
        sys.path.insert(0, chakra_dir)

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder.
    # All intermediate files go to _TMP_DIR (node-local storage) to avoid
    # Lustre I/O latency on per-wave trace + protobuf writes.
    output_name = workload_name if workload_name else file_name

    workload_dir = os.path.join(_TMP_DIR, 'workload', output_name)
    os.makedirs(workload_dir, exist_ok=True)

    input_path  = os.path.join(_TMP_DIR, 'trace', f'{file_name}.txt')
    output_path = os.path.join(_TMP_DIR, 'workload', output_name, 'llm')

    logger.debug("Generating graph: input=%s output=%s num_npus=%d npu_offset=%d",
                 input_path, output_path, num_npus, npu_offset,
                 extra={"node_id": node_id, "instance_id": instance_id})

    from chakra.src.converter.llm_converter import LLMConverter
    converter = LLMConverter(input_path, output_path, num_npus, npu_offset, enable_local_offloading)
    converter.convert()
