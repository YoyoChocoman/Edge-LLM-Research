#include "exp7-trace.h"
#include <algorithm>
#include <atomic>
#include <fstream>
#include <mutex>
#include <vector>

struct Event {
    int run, step, before, kind;
    size_t count;
    int64_t values[10];
};

static std::vector<Event> events;
static std::mutex mutex;
static std::atomic<bool> active{false};
static int current_run, current_step, current_before;
static size_t dropped = 0;

void exp7_trace_reset(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex);
    active = false;
    events.clear();
    events.reserve(capacity);
    dropped = 0;
}

void exp7_trace_begin(int run, int step, int before) {
    std::lock_guard<std::mutex> lock(mutex);
    current_run = run;
    current_step = step;
    current_before = before;
    active = true;
}

void exp7_trace_end() {
    std::lock_guard<std::mutex> lock(mutex);
    active = false;
}

void exp7_trace_record(int kind, const int64_t * values, size_t count) {
    if (!active.load()) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (!active.load()) return;
    // Never grow the buffer in a measured step.
    if (events.size() == events.capacity() || count > 10) {
        ++dropped;
        return;
    }
    Event event{current_run, current_step, current_before, kind, count, {}};
    std::copy(values, values + count, event.values);
    events.push_back(event);
}

bool exp7_trace_write(const char * path) {
    static const char * kinds[] = {"kv", "mask_build", "mask_check", "graph", "cuda", "padding"};
    static const char * fields[][10] = {
        {"n_kv", "kv_capacity", "n_tokens"},
        {"ne0", "ne1", "ne2", "ne3", "dtype"},
        {"old0", "old1", "old2", "old3", "new0", "new1", "new2", "new3", "compatible"},
        {"reused", "reuse_disabled", "n_tokens"},
        {"device", "graph_uid", "enabled", "compatible", "properties_changed",
         "warmup_before", "warmup_after", "use_graph", "capture", "instance_before"},
        {"required_padding", "effective_padding", "n_kv"}
    };
    std::ofstream out(path);
    out << "{\n  \"dropped_events\": " << dropped << ",\n  \"events\": [\n";
    for (size_t i = 0; i < events.size(); ++i) {
        const auto & e = events[i];
        out << "    {\"run_id\": " << e.run << ", \"decode_step\": " << e.step
            << ", \"n_past_before_eval\": " << e.before << ", \"kind\": \"" << kinds[e.kind] << "\"";
        for (size_t j = 0; j < e.count; ++j) {
            out << ", \"" << fields[e.kind][j] << "\": " << e.values[j];
        }
        out << "}" << (i + 1 < events.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    out.close();
    return bool(out);
}
