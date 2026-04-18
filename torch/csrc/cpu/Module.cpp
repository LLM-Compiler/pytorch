#include <ATen/cpu/Utils.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/numa.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cpu {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  cpu.def("_is_avx2_supported", at::cpu::is_avx2_supported);
  cpu.def("_is_avx512_supported", at::cpu::is_avx512_supported);
  cpu.def("_is_avx512_vnni_supported", at::cpu::is_avx512_vnni_supported);
  cpu.def("_is_avx512_bf16_supported", at::cpu::is_avx512_bf16_supported);
  cpu.def("_is_amx_tile_supported", at::cpu::is_amx_tile_supported);
  cpu.def("_init_amx", at::cpu::init_amx);
  cpu.def("_L1d_cache_size", at::cpu::L1d_cache_size);
  cpu.def("_L2_cache_size", at::cpu::L2_cache_size);
  cpu.def("_set_numa_enabled", [](bool enabled) {
    FLAGS_caffe2_cpu_numa_enabled = enabled;
  });
  cpu.def("_is_numa_enabled", []() {
    return c10::IsNUMAEnabled();
  });
  cpu.def("_num_numa_nodes", []() {
    return c10::GetNumNUMANodes();
  });
  // Move the pages backing a dense CPU tensor's storage to a specific NUMA
  // node in-place. This is the primitive used by NUMA-aware allocators to
  // place freshly-allocated (uninitialized) buffers directly on the target
  // socket, avoiding the `clone + copy` detour when the caller will overwrite
  // the data anyway (e.g. MM kernel outputs).
  cpu.def("_numa_move", [](const at::Tensor& tensor, int64_t numa_node_id) {
    TORCH_CHECK(
        tensor.device().is_cpu() && tensor.layout() == c10::kStrided,
        "_numa_move is only supported for dense CPU tensors");
    TORCH_CHECK(
        c10::IsNUMAEnabled(),
        "_numa_move requires NUMA support enabled at build/runtime");
    const auto num_numa_nodes = c10::GetNumNUMANodes();
    TORCH_CHECK(
        num_numa_nodes > 0,
        "_numa_move could not determine NUMA node count");
    TORCH_CHECK(
        numa_node_id >= 0 && numa_node_id < num_numa_nodes,
        "_numa_move expected numa_node in [0, ",
        num_numa_nodes - 1,
        "], but got ",
        numa_node_id);
    auto storage_data_ptr = tensor.storage().data_ptr().get();
    auto storage_nbytes = tensor.storage().nbytes();
    if (storage_data_ptr != nullptr && storage_nbytes > 0) {
      c10::NUMAMove(
          storage_data_ptr, storage_nbytes, static_cast<int>(numa_node_id));
    }
  });
}

} // namespace torch::cpu
