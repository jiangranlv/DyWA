#include <torch/extension.h>

torch::Tensor ur5_ik_cuda_forward(
        torch::Tensor T,
        torch::Tensor q_ref,
        torch::Tensor q_out);
// torch::Tensor ur5_ik_cuda_backward(torch::Tensor T);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ur5_ik_forward(torch::Tensor T, torch::Tensor q_ref, torch::Tensor q_out) {
    CHECK_INPUT(T);
    CHECK_CUDA(q_ref);
    CHECK_CUDA(q_out);
    return ur5_ik_cuda_forward(T, q_ref, q_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ur5_ik_forward, "UR5 inverse kinematics forward, with seed (CUDA)");
    // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
