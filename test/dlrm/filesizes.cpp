#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
using float32_t = float;
using float16_t = _Float16;
bool read_buff(const char* filename, size_t* bytes)
{
    int fd = open(filename, O_RDONLY);
    if(fd < 0)
    {
        std::cout << "Invalid file " << filename << std::endl;
        return false;
    }
    struct stat stats;

    fstat(fd, &stats);
    *bytes = stats.st_size;

    std::cout << "Read " << filename << " bytes " << *bytes << std::endl;

    close(fd);
    return true;
}

int main()
{

    std::string input, output_ref, grad_ref, upstream_grad, bottom_mlp_grad_ref;

    int         fp     = 16;
    std::string fp_str = std::to_string(fp);
    size_t      bytes16[5], bytes32[5];

    input               = "data/input_fp" + fp_str;
    output_ref          = "data/output_fp" + fp_str;
    upstream_grad       = "data/input_grad_fp" + fp_str;
    grad_ref            = "data/output_input_grad_fp" + fp_str;
    bottom_mlp_grad_ref = "data/output_mlp_input_grad_fp" + fp_str;

    std::cout << "float_16 sizes: \n";
    read_buff(input.c_str(), &bytes16[0]);
    read_buff(output_ref.c_str(), &bytes16[1]);
    read_buff(upstream_grad.c_str(), &bytes16[2]);
    read_buff(grad_ref.c_str(), &bytes16[3]);
    read_buff(bottom_mlp_grad_ref.c_str(), &bytes16[4]);

    fp_str              = std::to_string(32);
    input               = "data/input_fp" + fp_str;
    output_ref          = "data/output_fp" + fp_str;
    upstream_grad       = "data/input_grad_fp" + fp_str;
    grad_ref            = "data/output_input_grad_fp" + fp_str;
    bottom_mlp_grad_ref = "data/output_mlp_input_grad_fp" + fp_str;

    std::cout << "float_32 sizes: \n";
    read_buff(input.c_str(), &bytes32[0]);
    read_buff(output_ref.c_str(), &bytes32[1]);
    read_buff(upstream_grad.c_str(), &bytes32[2]);
    read_buff(grad_ref.c_str(), &bytes32[3]);
    read_buff(bottom_mlp_grad_ref.c_str(), &bytes32[4]);

    std::cout << "\nnumber of elements: \n"
              << "input: " << bytes16[0] / sizeof(float16_t) << ", "
              << bytes32[0] / sizeof(float32_t) << std::endl
              << "output_ref: " << bytes16[1] / sizeof(float16_t) << ", "
              << bytes32[1] / sizeof(float32_t) << std::endl
              << "upstream_grad: " << bytes16[2] / sizeof(float16_t) << ", "
              << bytes32[2] / sizeof(float32_t) << std::endl
              << "grad_ref: " << bytes16[3] / sizeof(float16_t) << ", "
              << bytes32[3] / sizeof(float32_t) << std::endl
              << "bottom_mlp_grad_ref: " << bytes16[4] / sizeof(float16_t) << ", "
              << bytes32[4] / sizeof(float32_t) << std::endl;
}
