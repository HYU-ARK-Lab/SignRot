#pragma once

#include <c10/cuda/CUDAStream.h>

// Every kernel in this extension must launch on PyTorch's current stream.
// Launching on the legacy default stream (the `<<<grid, block>>>` default)
// silently escapes CUDA graph capture -- the captured graph comes out empty --
// and breaks stream ordering against the rest of the model.
#define SIGNROT_STREAM (c10::cuda::getCurrentCUDAStream().stream())
