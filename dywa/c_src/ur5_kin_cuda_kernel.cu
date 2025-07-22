#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>


namespace {
    template <typename scalar_t>
        __device__ __inline__ int SIGN(const scalar_t x) {
            return (x > 0) - (x < 0);
        }

    template <typename scalar_t>
        __device__ __inline__ scalar_t ANGD(const scalar_t x, const scalar_t y) {
            const scalar_t d = (x > y) ? (x-y) : (y-x);
            const scalar_t m = fmod(d, 2 * M_PI); // 0 ~ 2*PI
            return m > M_PI? 2*M_PI - m : m;
        }

    template <typename scalar_t>
        __device__ __inline__ void Q1(
                const scalar_t T02,
                const scalar_t T03,
                const scalar_t T12,
                const scalar_t T13,
                scalar_t* const __restrict__ q1){
            ////////////////////////////// shoulder rotate joint (q1)
            /////////////////////////////////
            //
            static constexpr scalar_t d4 = 0.10915;
            static constexpr scalar_t d6 = 0.0823;
            static constexpr scalar_t ZERO_THRESH = 1e-8;
            static constexpr scalar_t PI = M_PI;

            const scalar_t A = d6 * T12 - T13;
            const scalar_t B = d6 * T02 - T03;
            const scalar_t R = A * A + B * B;
            if (fabs(A) < ZERO_THRESH) {
                scalar_t div;
                if (fabs(fabs(d4) - fabs(B)) < ZERO_THRESH)
                    div = -SIGN(d4) * SIGN(B);
                else
                    div = -d4 / B;
                scalar_t arcsin = asin(div);
                if (fabs(arcsin) < ZERO_THRESH) arcsin = 0.0;
                if (arcsin < 0.0)
                    q1[0] = arcsin + 2.0 * PI;
                else
                    q1[0] = arcsin;
                q1[1] = PI - arcsin;
            } else if (fabs(B) < ZERO_THRESH) {
                scalar_t div;
                if (fabs(fabs(d4) - fabs(A)) < ZERO_THRESH)
                    div = SIGN(d4) * SIGN(A);
                else
                    div = d4 / A;
                const scalar_t arccos = acos(div);
                q1[0] = arccos;
                q1[1] = 2.0 * PI - arccos;
            } else if (d4 * d4 > R) {
                return;
            } else {
                const scalar_t arccos = acos(d4 / sqrt(R));
                const scalar_t arctan = atan2(-B, A);
                scalar_t pos = arccos + arctan;
                scalar_t neg = -arccos + arctan;
                if (fabs(pos) < ZERO_THRESH) pos = 0.0;
                if (fabs(neg) < ZERO_THRESH) neg = 0.0;
                if (pos >= 0.0)
                    q1[0] = pos;
                else
                    q1[0] = 2.0 * PI + pos;
                if (neg >= 0.0)
                    q1[1] = neg;
                else
                    q1[1] = 2.0 * PI + neg;
            }
        }

    template <typename scalar_t>
        __device__ __inline__ void Q5(
                const scalar_t* const __restrict__ q1,
                const scalar_t T03,
                const scalar_t T13,
                scalar_t* const __restrict__ q5){
            ////////////////////////////// wrist 2 joint (q5)
            /////////////////////////////////
            static constexpr scalar_t d4 = 0.10915;
            static constexpr scalar_t ZERO_THRESH = 1e-8;
            static constexpr scalar_t d6 = 0.0823;
            static constexpr scalar_t PI = M_PI;
            for (int i = 0; i < 2; ++i) {
                scalar_t numer = (T03 * sin(q1[i]) - T13 * cos(q1[i]) - d4);
                scalar_t div;
                if (fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
                    div = SIGN(numer) * SIGN(d6);
                else
                    div = numer / d6;
                scalar_t arccos = acos(div);
                q5[i*2+0] = arccos;
                q5[i*2+1] = 2.0 * PI - arccos;
            }
        }


    template <typename scalar_t>
        __global__ void ur5_ik_cuda_forward_kernel(
                // const scalar_t* __restrict__ Ts,
                // const scalar_t* __restrict__ q0s,
                // scalar_t* __restrict__ qs,
                const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Ts,
                const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> q0s,
                torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> qs,
                const size_t n) {

            // UR5 parameters.
            static constexpr scalar_t d1 = 0.089159;
            static constexpr scalar_t a2 = -0.42500;
            static constexpr scalar_t a3 = -0.39225;
            static constexpr scalar_t d4 = 0.10915;
            static constexpr scalar_t d5 = 0.09465;
            static constexpr scalar_t d6 = 0.0823;
            static constexpr scalar_t PI = M_PI;
            static constexpr scalar_t ZERO_THRESH = 0.00000001;
            static constexpr scalar_t q6_des = 0.0;

            const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n){
                return;
            }

            // const scalar_t* const T = Ts + index * 16;
            // const scalar_t* const qref = q0s + index * 6;
            // scalar_t* const q_out = qs + index * 6;
            //
            const auto T = Ts[index];
            const auto qref = q0s[index];
            auto q_out = qs[index];

            // const scalar_t qref_0 = q0s[index*6 +0];
            // const scalar_t qref_1 = q0s[index*6 +1];
            // const scalar_t qref_2 = q0s[index*6 +2];
            // const scalar_t qref_3 = q0s[index*6 +3];
            // const scalar_t qref_4 = q0s[index*6 +4];
            // const scalar_t qref_5 = q0s[index*6 +5];

            // scalar_t* const q = qs + index * 8 * 6;
            // scalar_t* const q_out = qs + index * 6;

            // Enable all floating point exceptions but FE_INEXACT
            // To use this, #include <fenv.h>
            // feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

            const scalar_t T02 = -T[0];
            const scalar_t T00 = T[1];
            const scalar_t T01 = T[2];
            const scalar_t T03 = -T[3];
            const scalar_t T12 = -T[4];
            const scalar_t T10 = T[5];
            const scalar_t T11 = T[6];
            const scalar_t T13 = -T[7];
            const scalar_t T22 = T[8];
            const scalar_t T20 = -T[9];
            const scalar_t T21 = -T[10];
            const scalar_t T23 = T[11];

            scalar_t q1[2];
            Q1(T02, T03, T12, T13, q1);

            scalar_t q5[4];
            Q5(q1, T03, T13, q5);

            scalar_t d_best = 24 * PI;
            for(int ii=0; ii < 4; ++ii){
                const int i = ii / 2;
                scalar_t c1 = cos(q1[i]), s1 = sin(q1[i]);
                scalar_t c5 = cos(q5[ii]), s5 = sin(q5[ii]);
                scalar_t q6;
                ////////////////////////////// wrist 3 joint (q6)
                /////////////////////////////////
                if (fabs(s5) < ZERO_THRESH)
                    q6 = q6_des;
                else {
                    q6 = atan2(SIGN(s5) * -(T01 * s1 - T11 * c1),
                            SIGN(s5) * (T00 * s1 - T10 * c1));
                    if (fabs(q6) < ZERO_THRESH) q6 = 0.0;
                    if (q6 < 0.0) q6 += 2.0 * PI;
                }
                ////////////////////////////////////////////////////////////////////////////////

                scalar_t q2[2], q3[2], q4[2];
                ///////////////////////////// RRR joints (q2,q3,q4)
                ///////////////////////////////
                scalar_t c6 = cos(q6), s6 = sin(q6);
                const scalar_t k0 = (T00 * c1 + T10 * s1);
                const scalar_t k1 = (T01 * c1 + T11 * s1);
                const scalar_t k2 = (T02 * c1 + T12 * s1);
                scalar_t x04x =
                    -s5 * k2 -
                    c5 * (s6 * k1 - c6 * k0);
                scalar_t x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5;
                scalar_t p13x =
                    d5 * (s6 * k0 + c6 * k1) -
                    d6 * k2 + T03 * c1 + T13 * s1;
                scalar_t p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6);

                scalar_t c3 =
                    (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3);
                if (fabs(fabs(c3) - 1.0) < ZERO_THRESH)
                    c3 = SIGN(c3);
                else if (fabs(c3) > 1.0) {
                    continue;
                }
                scalar_t arccos = acos(c3);
                q3[0] = arccos;
                q3[1] = 2.0 * PI - arccos;
#if 0
                scalar_t denom = a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3;
                scalar_t s3 = sin(arccos);
                scalar_t A = (a2 + a3 * c3), B = a3 * s3;
                q2[0] =
                    atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom);
                q2[1] =
                    atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom);
#elif 0
                scalar_t s3 = sin(arccos);
                scalar_t A = (a2 + a3 * c3), B = a3 * s3;
                q2[0] =
                    atan2(A * p13y - B * p13x, A * p13x + B * p13y);
                q2[1] =
                    atan2(A * p13y + B * p13x, A * p13x - B * p13y);
#else
                const scalar_t B_A = (a3 * sin(arccos)) / (a2 + a3 * c3);
                q2[0] = atan2(-p13y + B_A * p13x, -p13x - B_A * p13y);
                q2[1] = atan2(-p13y - B_A * p13x, -p13x + B_A * p13y);
#endif
                scalar_t c23_0 = cos(q2[0] + q3[0]);
                scalar_t s23_0 = sin(q2[0] + q3[0]);
                scalar_t c23_1 = cos(q2[1] + q3[1]);
                scalar_t s23_1 = sin(q2[1] + q3[1]);
                q4[0] = atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0);
                q4[1] = atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1);
                ////////////////////////////////////////////////////////////////////////////////
                for (int k = 0; k < 2; ++k) {
                    if (fabs(q2[k]) < ZERO_THRESH)
                        q2[k] = 0.0;
                    else if (q2[k] < 0.0) {
                        q2[k] += 2.0 * PI;
                    }
                    if (fabs(q4[k]) < ZERO_THRESH)
                        q4[k] = 0.0;
                    else if (q4[k] < 0.0)
                        q4[k] += 2.0 * PI;

                    const scalar_t dq = (
                            ANGD(q1[i], qref[0]) +
                            ANGD(q2[k], qref[1]) +
                            ANGD(q3[k], qref[2]) +
                            ANGD(q4[k], qref[3]) +
                            ANGD(q5[ii], qref[4]) +
                            ANGD(q6, qref[5])
                            );

                    if (!isnan(dq) && dq < d_best){
                        q_out[0] = q1[i];
                        q_out[1] = q2[k];
                        q_out[2] = q3[k];
                        q_out[3] = q4[k];
                        q_out[4] = q5[ii];
                        q_out[5] = q6;
                        d_best = dq;
                    }
                }
            }

            // q_out[0] = 0.0;
            // q_out[1] = 0.0;
            // q_out[2] = 0.0;
            // q_out[3] = 0.0;
            // q_out[4] = 0.0;
            // q_out[5] = 0.0;

            // __syncthreads();
            // q_out[0] = sol[0];
            // qs[0] = 1.2; // ok
            // qs[0] = sol[0]; // ok?
            // qs[index*6+0] = sol[0];
            // q_out[1] = sol[1];
            // q_out[2] = sol[2];
            // q_out[3] = sol[3];
            // q_out[4] = sol[4];
            // q_out[5] = sol[5];
            return;
        }
}

torch::Tensor ur5_ik_cuda_forward(
        torch::Tensor T,
        torch::Tensor q_ref,
        torch::Tensor q_out
        ){
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // j
    at::DeviceGuard guard(T.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto batch_size = T.size(0);

    const int threads = 512;
    const dim3 blocks( (batch_size + threads - 1) / threads);
    // const dim3 blocks(1);

    // Create output tensor.
    // max num sols = 8
    // auto opts = (torch::TensorOptions()
    //         .dtype(torch::kFloat32)
    //         .layout(torch::kStrided)
    //         .device(T.device()));
    // auto q_out = torch::zeros({batch_size, 6}, opts);
    // auto q_out = torch::zeros({batch_size, 6}, T.options());
    const auto T2 = T.reshape({-1,16});

    AT_DISPATCH_FLOATING_TYPES(T.type(), "ur5_ik_cuda_forward", ([&] {
                ur5_ik_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        // T.data<scalar_t>(),
                        // q_ref.data<scalar_t>(),
                        // q_out.data<scalar_t>(),

                        T2.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        q_ref.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        q_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        batch_size
                        );
                }));
    return q_out;
}
