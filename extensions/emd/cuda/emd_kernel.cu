#ifndef _EMD_KERNEL
#define _EMD_KERNEL

#include <vector>
#include <torch/extension.h>

void emd_forward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    at::Tensor dist, 
    at::Tensor assignment);

void emd_backward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor match, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2);

#endif