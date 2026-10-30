#pragma once

#include <cuda_runtime.h>
#include <thread>

namespace bench {
// Return success or the first real error; only not-ready means keep waiting.
template <typename Query>
inline cudaError_t wait_query(Query query, bool yield) {
    cudaError_t status;
    while ((status = query()) == cudaErrorNotReady) {
        if (yield) std::this_thread::yield();
    }
    return status;
}

inline cudaError_t wait_stream(cudaStream_t stream, bool query, bool yield) {
    if (!query) return cudaStreamSynchronize(stream);
    return wait_query([&] { return cudaStreamQuery(stream); }, yield);
}

inline cudaError_t wait_event(cudaEvent_t event, bool query, bool yield) {
    if (!query) return cudaEventSynchronize(event);
    return wait_query([&] { return cudaEventQuery(event); }, yield);
}
} // namespace bench
