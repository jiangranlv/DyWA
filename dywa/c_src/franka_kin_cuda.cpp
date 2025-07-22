#include <torch/extension.h>

torch::Tensor franka_ik_cuda_forward(
        torch::Tensor T,
        // torch::Tensor q_ref,
        torch::Tensor q_out,
        const float width,
        const int iter
        );
// torch::Tensor franka_ik_cuda_backward(torch::Tensor T);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor franka_ik_forward(torch::Tensor T,
        // torch::Tensor q_ref,
        torch::Tensor q_out,
        const float width,
        const int iter
        ) {
    CHECK_INPUT(T);
    // CHECK_CUDA(q_ref);
    CHECK_CUDA(q_out);
    // return franka_ik_cuda_forward(T, q_ref, q_out);
    return franka_ik_cuda_forward(T, q_out, width, iter);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &franka_ik_forward, "Franka inverse kinematics forward, with seed (CUDA)");
    // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
