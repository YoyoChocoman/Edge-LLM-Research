#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>

enum Exp6Kind { EXP6_KV, EXP6_MASK_BUILD, EXP6_MASK_CHECK, EXP6_GRAPH, EXP6_CUDA };

extern "C" {
GGML_API void exp6_trace_reset(size_t capacity);
GGML_API void exp6_trace_begin(int run, int step, int before);
GGML_API void exp6_trace_end();
GGML_API void exp6_trace_record(int kind, const int64_t * values, size_t count);
GGML_API bool exp6_trace_write(const char * path);
}

inline void exp6_record(Exp6Kind kind, std::initializer_list<int64_t> values) {
    exp6_trace_record(kind, values.begin(), values.size());
}
