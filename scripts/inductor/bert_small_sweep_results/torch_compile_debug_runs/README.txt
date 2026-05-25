# Per-configuration Torch compile debug (BERT multi-size sweep)
# Each subfolder name encodes: M<model>_B<batch>_S<seq>_<standard|numa_tp>
# With TORCH_COMPILE_DEBUG=1, Dynamo writes to:
#   <that_subfolder>/torch_compile_debug/run_YYYY_mm_dd_...-pid_*/
# The parent script restores env after each config.
