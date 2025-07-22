#include <torch/extension.h>

#ifdef EIGEN_NO_CUDA
#undef EIGEN_NO_CUDA
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <Eigen/Dense>


namespace {


    template<typename scalar_t>
        __device__ __inline__ scalar_t joint_limit_cost(const scalar_t* q)
        {
            // joint 0 : -2.8973 ~ 2.8973 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_0 = abs(q[ 0 ]/2.8973);

            // joint 1 : -1.7628 ~ 1.7628 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_1 = abs(q[ 1 ]/1.7628);

            // joint 2 : -1.7628 ~ 1.7628 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_2 = abs(q[ 2 ]/1.7628);

            // joint 3 : -3.0718 ~ -0.0698 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_3 = abs((q[ 3 ]+0.0698+1.5)/1.5);

            // joint 4 : -2.8973 ~ 2.8973 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_4 = abs(q[ 4 ]/2.8973);

            // joint 5 : -0.0175 ~ 3.7525 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_5 = abs((q[ 5 ]+0.0175-1.885)/1.885);

            // joint 6 : -2.8973 ~ 2.8973 -> -1 ~ 1 -> 0 ~ 1
            const scalar_t joint_6 = abs(q[ 6 ]/2.8973);

            // scalar_t max = std::max({joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6});
            const scalar_t max = joint_0*100 + joint_1 + joint_2 + joint_3 + joint_4 + joint_5 + joint_6;
            return max; // low is better
        }

#if 1
    template<typename scalar_t>
        __device__ void franka_IK_EE (
                const Eigen::Matrix<scalar_t,3,3> &R_EE,
                const Eigen::Matrix<scalar_t,3,1> &p_EE,
                const scalar_t q7,
                const Eigen::Matrix<scalar_t, 7, 1> &q_actual,
                Eigen::Matrix<scalar_t, 7, 4>& q_out)
        {
            // typedef Vector3 Vector3;
            // typedef Matrix3 Matrix3;
            using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
            using Matrix3 = Eigen::Matrix<scalar_t, 3, 3>;
            // const std::array< std::array<scalar_t, 7>, 4 > q_all_NAN = {{ {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
            //     {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
            //     {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
            //     {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}} }};
            q_out = Eigen::Matrix<scalar_t,7, 4>::Constant(NAN);
            const Eigen::Matrix<scalar_t, 7, 1> q_NAN = Eigen::Matrix<scalar_t,7,1>::Constant(NAN);
            Eigen::Matrix<scalar_t, 7, 4> q_all = Eigen::Matrix<scalar_t,7,4>::Constant(NAN);

            // Eigen::Map< Eigen::Matrix<scalar_t, 4, 4> > O_T_EE(O_T_EE_array.data());

            static constexpr const scalar_t d1 = 0.3330;
            static constexpr const scalar_t d3 = 0.3160;
            static constexpr const scalar_t d5 = 0.3840;
            static constexpr const scalar_t d7e = 0.2104;
            static constexpr const scalar_t a4 = 0.0825;
            static constexpr const scalar_t a7 = 0.0880;
            static constexpr const scalar_t LL24 = 0.10666225; // a4^2 + d3^2
            static constexpr const scalar_t LL46 = 0.15426225; // a4^2 + d5^2
            static constexpr const scalar_t L24 = 0.326591870689; // sqrt(LL24)
            static constexpr const scalar_t L46 = 0.392762332715; // sqrt(LL46)
            static constexpr const scalar_t thetaH46 = 1.35916951803; // atan(d5/a4);
            static constexpr const scalar_t theta342 = 1.31542071191; // atan(d3/a4);
            static constexpr const scalar_t theta46H = 0.211626808766; // acot(d5/a4);
                                                                       //

            const scalar_t q_min[7]{-2.8973, -1.7628, -2.8973,
                -3.0718, -2.8973, -0.0175, -2.8973};
            const scalar_t q_max[7]{2.8973, 1.7628, 2.8973,
                -0.0698, 2.8973, 3.7525, 2.8973};

            if (q7 <= q_min[6] || q7 >= q_max[6]){
                // printf("fail-A\n");
                return;
            }
            else{
                q_all.row(6).array() = q7;
                // for (int i = 0; i < 4; i++){
                //     q_all(6, i) = q7;
                // }

            }

            // compute p_6
            // Eigen::Matrix3d R_EE = O_T_EE.topLeftCorner<3, 3>();
            // const Vector3& z_EE = R_EE.template block<3, 1>(0, 2);
            const Vector3 z_EE = R_EE.col(2);
            // printf("z_EE = %f %f %f\n", z_EE[0], z_EE[1], z_EE[2]);
            // Eigen::Vector3d p_EE = O_T_EE.block<3, 1>(0, 3);
            const Vector3 p_7 = p_EE - d7e*z_EE;
            // printf("p_7 = %f %f %f\n", p_7[0], p_7[1], p_7[2]);

            // const Vector3 x_EE_6{cos(q7 - M_PI_4), -sin(q7 - M_PI_4), 0.0};
            const Vector3 x_EE_6{cos(q7 - M_PI_4), -sin(q7 - M_PI_4), 0.0};
            // printf("x_EE_6 = %f %f %f\n", x_EE_6[0], x_EE_6[1], x_EE_6[2]);
            const Vector3 x_6 = R_EE*x_EE_6; // disallowed
                                             // printf("x6(pre) = %f %f %f\n", x_6[0], x_6[1], x_6[2]);
                                             // x_6 /= x_6.norm(); // visibly increases accuracy
                                             // printf("x6 = %f %f %f\n", x_6[0], x_6[1], x_6[2]);

                                             // const Vector3 tmp = a7 * x_6;
                                             // const Vector3 tmp{a7 * x_6[0], a7 * x_6[1], a7 * x_6[2]};
                                             // const scalar_t aa = a7 * x_6[0];
                                             // const scalar_t bb = a7 * x_6[1];
                                             // const scalar_t cc = a7 * x_6[2];
            const Vector3 p_6 = p_7 - a7*x_6;
            // printf("a7 = %f \n", a7);
            // printf("x6 = %f %f %f\n", x_6[0], x_6[1], x_6[2]);
            // printf("tmp = %f %f %f\n", tmp[0], tmp[1], tmp[2]);
            // printf("alt = %f %f %f\n", aa, bb, cc);
            // printf("p7 = %f %f %f\n", p_7[0], p_7[1], p_7[2]);
            // printf("p6 = %f %f %f\n", p_6[0], p_6[1], p_6[2]);

            // compute q4
            const Vector3 p_2{0.0, 0.0, d1};
            // p_2 << 0.0, 0.0, d1;

            const Vector3 V26 = p_6 - p_2;
            // printf("V26 = %f %f %f\n", V26[0], V26[1], V26[2]);

            const scalar_t LL26 = V26[0]*V26[0] + V26[1]*V26[1] + V26[2]*V26[2];
            const scalar_t L26 = sqrt(LL26);
            // printf("L26 = %f %f\n", L26, LL26);

            if (L24 + L46 < L26 || L24 + L26 < L46 || L26 + L46 < L24){
                // printf("fail-B\n");
                return;
            }

            scalar_t theta246 = acos((LL24 + LL46 - LL26)/2.0/L24/L46);
            scalar_t q4 = theta246 + thetaH46 + theta342 - 2.0*M_PI;
            if (q4 <= q_min[3] || q4 >= q_max[3]){
                // printf("fail-C\n");
                return;
            }
            else{
                for (int i = 0; i < 4; i++)
                    q_all(3,i) = q4;
            }

            // compute q6
            scalar_t theta462 = acos((LL26 + LL46 - LL24)/2.0/L26/L46);
            scalar_t theta26H = theta46H + theta462;
            // printf("L26 theta26H = %f %f\n", L26, theta26H);
            scalar_t D26 = -L26*cos(theta26H); // NAN

            Vector3 Z_6 = z_EE.cross(x_6);
            Vector3 Y_6 = Z_6.cross(x_6);
            Matrix3 R_6;
            R_6.col(0) = x_6;
            R_6.col(1) = Y_6/Y_6.norm();
            R_6.col(2) = Z_6/Z_6.norm();
            Vector3 V_6_62 = R_6.transpose()*(-V26); // Zero ? disallowed

            scalar_t Phi6 = atan2(V_6_62[1], V_6_62[0]);
            // printf("D26 V60 V61 = %f %f %f\n", D26, V_6_62[0], V_6_62[1]);
            scalar_t Theta6 = asin(D26/sqrt(V_6_62[0]*V_6_62[0] + V_6_62[1]*V_6_62[1]));
            // printf("T6 P6 = %f %f\n", Theta6, Phi6);

            scalar_t q6[2];
            q6[0] = M_PI - Theta6 - Phi6;
            q6[1] = Theta6 - Phi6;

            for (int i = 0; i < 2; i++)
            {
                // printf("q6[i] = %f\n", q6[i]); // nan
                if (q6[i] <= q_min[5])
                    q6[i] += 2.0*M_PI;
                else if (q6[i] >= q_max[5])
                    q6[i] -= 2.0*M_PI;

                if (q6[i] <= q_min[5] || q6[i] >= q_max[5])
                {
                    q_all.col( 2*i ) = q_NAN;
                    q_all.col( 2*i + 1 ) = q_NAN;
                }
                else
                {
                    q_all(5,2*i) = q6[i];
                    q_all(5,2*i + 1) = q6[i];
                }
            }
            if (isnan(q_all(5, 0)) && isnan(q_all(5, 2))){
                // printf("fail-D\n");
                return;
            }

            // compute q1 & q2
            scalar_t thetaP26 = 3.0*M_PI_2 - theta462 - theta246 - theta342;
            scalar_t thetaP = M_PI - thetaP26 - theta26H;
            scalar_t LP6 = L26*sin(thetaP26)/sin(thetaP);

            Eigen::Matrix<scalar_t, 3, 4> z_5_all;
            Eigen::Matrix<scalar_t, 3, 4> V2P_all;

            for (int i = 0; i < 2; i++)
            {
                Vector3 z_6_5;
                z_6_5 << sin(q6[i]), cos(q6[i]), 0.0;
                Vector3 z_5 = R_6*z_6_5;
                Vector3 V2P = p_6 - LP6*z_5 - p_2;

                z_5_all.col( 2*i ) = z_5;
                z_5_all.col( 2*i + 1 ) = z_5;
                V2P_all.col( 2*i ) = V2P;
                V2P_all.col( 2*i + 1 ) = V2P;

                scalar_t L2P = V2P.norm();

                if (fabs(V2P[2]/L2P) > 0.999)
                {
                    q_all(0,2*i) = q_actual(0);
                    q_all(1,2*i) = 0.0;
                    q_all(0,2*i + 1) = q_actual(0);
                    q_all(1,2*i + 1) = 0.0;
                }
                else
                {
                    q_all(0,2*i) = atan2(V2P[1], V2P[0]);
                    q_all(1,2*i) = acos(V2P[2]/L2P);
                    if (q_all(0,2*i) < 0)
                        q_all(0,2*i + 1) = q_all(0,2*i) + M_PI;
                    else
                        q_all(0,2*i + 1) = q_all(0,2*i) - M_PI;
                    q_all(1,2*i + 1) = -q_all(1,2*i);
                }
            }

            for (int i = 0; i < 4; i++)
            {
                if ( q_all(0,i) <= q_min[0] || q_all(0,i) >= q_max[0]
                        || q_all(1,i) <= q_min[1] || q_all(1,i) >= q_max[1] )
                {
                    q_all.col( i ) = q_NAN;
                    continue;
                }

                // compute q3
                Vector3 z_3 = V2P_all.col( i )/V2P_all.col( i ).norm();
                Vector3 Y_3 = -V26.cross(V2P_all.col( i ));
                Vector3 y_3 = Y_3/Y_3.norm();
                Vector3 x_3 = y_3.cross(z_3);
                Matrix3 R_1;
                scalar_t c1 = cos(q_all(0,i));
                scalar_t s1 = sin(q_all(0,i));
                R_1 <<   c1,  -s1,  0.0,
                    s1,   c1,  0.0,
                    0.0,  0.0,  1.0;
                Matrix3 R_1_2;
                scalar_t c2 = cos(q_all(1,i));
                scalar_t s2 = sin(q_all(1,i));
                R_1_2 <<   c2,  -s2, 0.0,
                      0.0,  0.0, 1.0,
                      -s2,  -c2, 0.0;
                Matrix3 R_2 = R_1*R_1_2; // disallowed
                Vector3 x_2_3 = R_2.transpose()*x_3; // disallowed
                q_all(2,i) = atan2(x_2_3[2], x_2_3[0]);

                if (q_all(2,i) <= q_min[2] || q_all(2,i) >= q_max[2])
                {
                    q_all.col( i ) = q_NAN;
                    continue;
                }

                // compute q5
                Vector3 VH4 = p_2 + d3*z_3 + a4*x_3 - p_6 + d5*z_5_all.col( i );
                Matrix3 R_5_6;
                scalar_t c6 = cos(q_all(5,i));
                scalar_t s6 = sin(q_all(5,i));
                R_5_6 <<   c6,  -s6,  0.0,
                      0.0,  0.0, -1.0,
                      s6,   c6,  0.0;
                Matrix3 R_5 = R_6*R_5_6.transpose(); // disallowed
                Vector3 V_5_H4 = R_5.transpose()*VH4;

                q_all(4,i) = -atan2(V_5_H4[1], V_5_H4[0]);
                if (q_all(4,i) <= q_min[4] || q_all(4,i) >= q_max[4])
                {
                    q_all.col( i ) = q_NAN;
                    continue;
                }
            }
            // printf("last\n");
            q_all.row(6).array() = q7;
            q_out = q_all;
        }
#endif

    template <typename scalar_t>
        __global__ void franka_ik_cuda_forward_kernel(
                const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> Ts,
                // const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> q0s,
                torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> qs,
                const size_t n,
                const scalar_t width = 0.08,
                const int num_iter = 1
                ) {

            const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n){
                return;
            }
            curandState localState;
            curand_init(1234, index, 0, &localState);

            // auto localState = state[id]; // why?
            // curend_init(1234, index, 0, &state)

            // Deal with current batch.
            const auto T = Ts[index];
            // const auto q_ref = q0s[index];
            auto q_out = qs[index];

            // We're going to assume this offset application was
            // handled outside of the scope of this kernel.
            // Eigen::Quaterniond q(targetHandOrientation);
            // Eigen::Matrix3d targetRotation = q.normalized().toRotationMatrix();
            // Eigen::Vector3d offset(0.0, 0.0, 0.1034);
            // Eigen::Vector3d targetPosition = targetRotation * offset + targetHandPosition;
            scalar_t q[10]{}; // initializes to zero
            scalar_t q_best[10]{}; // initializes to zero
            scalar_t best_cost{1000000.0};

            q[9] = 0; // 0 means infeasible
            q_best[9] = 0;
            Eigen::Matrix<scalar_t, 7, 1> q_cur;
            q_cur << 0.0000, 0.0000, 0.0000,
                  -0.9425, 0.0000, 1.1205, 0.0000;

            // std::default_random_engine gen;
            // std::uniform_real_distribution<double> distribution1(-2., 2.);

            // scalar_t q_results[4][7];

            Eigen::Matrix<scalar_t, 3, 3> targetRotation;
            targetRotation << T[0], T[1], T[2],
            T[4], T[5], T[6],
            T[8], T[9], T[10];
            // printf("[R]\n %f %f %f\n %f %f %f\n %f %f %f\n",T[0], T[1], T[2],
            //         T[4], T[5], T[6],
            //         T[8], T[9], T[10]);

            Eigen::Matrix<scalar_t, 3, 1> targetPosition;
            targetPosition << T[3], T[7], T[11];
            // printf("[T]\n %f %f %f\n",T[3], T[7], T[11]);

            Eigen::Matrix<scalar_t, 7, 4> q_results;

            bool feasible{false};
            for (int x = 0; x < num_iter; x++)
            {
                // Compute IK.
                // const scalar_t q7{0.0};// = distribution1(gen);
                // const scalar_t q7 = 5.6 * curand_uniform(&localState) - 2.8;
                const scalar_t q7 = CUDART_PI_F * curand_uniform(&localState) - CUDART_PI_F/2;

                // const scalar_t q7 = 2 * M_PI * curand_uniform(&localState);
                franka_IK_EE<scalar_t>(
                        targetRotation,
                        targetPosition,
                        q7,
                        q_cur,
                        q_results);

                // Apply heuristics
                for (int i = 0; i < 4; i++)
                {
                    feasible = true;

                    // Check if solution is NaN.
                    for (int j = 0; j < 7; j++)
                    {
                        // if (isnan(q_results(j,i)))// || (q_results(j,i)==0.0))
                        if (isnan(q_results(j,i)) || (q_results(j,i)==0.0))
                        {
                            // printf("infeasible since nan at %d %d\n", i, j);
                            feasible=false;
                            break;
                        }
                    }

                    // Append solution.
                    if(feasible) {
                        for (int k = 0; k < 7; k++) {
                            q[ k ] = q_results(k,i);
                        }
                        q[ 7 ] = q[ 8 ] = width;
                        q[ 9 ] = 1;

                        const scalar_t cost = joint_limit_cost<scalar_t>(q);
                        // printf("%f \n", cost);

                        if(cost < best_cost) {
                            for (int k = 0; k < 10; k++) {
                                q_best[ k ] = q[ k ];
                            }
                            best_cost = cost;
                        }

                    }
                }
            }

            // OR to memcpy
            for(int i=0; i<10; ++i){
                q_out[i] = q_best[i];
            }
        }

}

torch::Tensor franka_ik_cuda_forward(
        torch::Tensor T,
        // torch::Tensor q_ref,
        torch::Tensor q_out,
        const float width,
        const int iter
        ){
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // j
    at::DeviceGuard guard(T.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto batch_size = T.size(0);

    const int threads = 256;
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
    const auto T_flat = T.reshape({-1,16});

    AT_DISPATCH_FLOATING_TYPES(T.type(), "franka_ik_cuda_forward", ([&] {
                franka_ik_cuda_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        T_flat.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        // q_ref.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        q_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                        batch_size,
                        width,
                        iter
                        );
                }));
    return q_out;
}
