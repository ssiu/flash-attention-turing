#include <torch/extension.h>

//torch::Tensor flash_fwd_v0(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v3(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v5(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v6(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v7(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v9(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v11(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v12(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v13(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v14(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v15(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

torch::Tensor flash_fwd_v17(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v18(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //m.def("flash_fwd_v0", &flash_fwd_v0, "flash fwd v0");
    //m.def("flash_fwd_v1", &flash_fwd_v1, "flash fwd v1");
    //m.def("flash_fwd_v2", &flash_fwd_v2, "flash fwd v2");
    //m.def("flash_fwd_v3", &flash_fwd_v3, "flash fwd v3");
    //m.def("flash_fwd_v4", &flash_fwd_v4, "flash fwd v4");
    //m.def("flash_fwd_v5", &flash_fwd_v5, "flash fwd v5");
    //m.def("flash_fwd_v6", &flash_fwd_v6, "flash fwd v6");
    //m.def("flash_fwd_v7", &flash_fwd_v7, "flash fwd v7");
    //m.def("flash_fwd_v8", &flash_fwd_v8, "flash fwd v8");
    //m.def("flash_fwd_v9", &flash_fwd_v9, "flash fwd v9");
    //m.def("flash_fwd_v11", &flash_fwd_v11, "flash fwd v11"); DOES NOT WORK
    //m.def("flash_fwd_v12", &flash_fwd_v12, "flash fwd v12");
    //m.def("flash_fwd_v13", &flash_fwd_v13, "flash fwd v13");
    //m.def("flash_fwd_v14", &flash_fwd_v14, "flash fwd v14");
    //m.def("flash_fwd_v15", &flash_fwd_v15, "flash fwd v15");
    //m.def("flash_fwd_v16", &flash_fwd_v16, "flash fwd v16");
    m.def("flash_fwd_v17", &flash_fwd_v17, "flash fwd v17");
    //m.def("flash_fwd_v18", &flash_fwd_v18, "flash fwd v18");
}