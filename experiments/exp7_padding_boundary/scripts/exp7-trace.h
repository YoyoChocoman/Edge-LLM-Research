#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>

enum Exp6Kind { EXP7_KV, EXP7_MASK_BUILD, EXP7_MASK_CHECK, EXP7_GRAPH, EXP7_CUDA, EXP7_PADDING };

extern "C" {
GGML_API void exp7_trace_reset(size_t capacity);
GGML_API void exp7_trace_begin(int run, int step, int before);
GGML_API void exp7_trace_end();
GGML_API void exp7_trace_record(int kind, const int64_t * values, size_t count);
GGML_API bool exp7_trace_write(const char * path);
}

inline void exp7_record(Exp6Kind kind, std::initializer_list<int64_t> values) {
    exp7_trace_record(kind, values.begin(), values.size());
}
