"""Insert observation hooks into the private llama.cpp source copy."""
from pathlib import Path
import shutil

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Runtime source does not match the expected version: {old[:100]!r}")
    return text.replace(old, new, 1)

def patch_runtime(source, padding):
    edits = {
        "ggml/src/CMakeLists.txt": [
            ("set_target_properties(ggml-base PROPERTIES", "target_sources(ggml-base PRIVATE exp7-trace.cpp)\n\nset_target_properties(ggml-base PROPERTIES"),
        ],
        "src/llama-kv-cache.cpp": [
            ("const uint32_t n_pad_cur = std::max(n_pad, 256u);", f"const uint32_t n_pad_cur = std::max(n_pad, {padding}u);"),
            ("    return result;\n}\n\nggml_tensor * llama_kv_cache::get_k(", "    exp7_record(EXP7_PADDING, {n_pad, n_pad_cur, result});\n    return result;\n}\n\nggml_tensor * llama_kv_cache::get_k("),
            ("    n_kv = kv->get_n_kv(sinfos[i_cur]);", "    n_kv = kv->get_n_kv(sinfos[i_cur]);\n    exp7_record(EXP7_KV, {n_kv, kv->get_size(), ubatches[i_cur].n_tokens});"),
        ],
        "src/llama-graph.cpp": [
            ('    ggml_set_name(res, "attn_inp_kq_mask");', '    ggml_set_name(res, "attn_inp_kq_mask");\n    exp7_record(EXP7_MASK_BUILD, {res->ne[0], res->ne[1], res->ne[2], res->ne[3], type});'),
            ("    res &= (kq_mask->ne[3] == n_stream);", "    res &= (kq_mask->ne[3] == n_stream);\n    exp7_record(EXP7_MASK_CHECK, {kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],\n        n_kv, n_tokens/n_stream, 1, n_stream, res});"),
        ],
        "src/llama-context.cpp": [
            ("    if (!graph_reuse_disable && res->can_reuse(gparams)) {", "    const bool exp7_reused = !graph_reuse_disable && res->can_reuse(gparams);\n    if (exp7_reused) {"),
            ("    // set the input data for the input tensors", "    exp7_record(EXP7_GRAPH, {exp7_reused, graph_reuse_disable, ubatch.n_tokens});\n\n    // set the input data for the input tensors"),
        ],
        "ggml/src/ggml-cuda/ggml-cuda.cu": [
            ("    const void * graph_key = nullptr;", "    const void * graph_key = nullptr;\n    int exp7_enabled = -1, exp7_compatible = -1, exp7_changed = -1;\n    int exp7_warmup_before = -1, exp7_warmup_after = -1, exp7_instance = -1;"),
            ("    if (graph->is_enabled()) {\n        const bool graph_compatible", "    exp7_enabled = graph->is_enabled();\n    exp7_warmup_before = graph->warmup_complete;\n    exp7_instance = graph->instance != nullptr;\n    if (graph->is_enabled()) {\n        const bool graph_compatible"),
            ("        if (graph_compatible) {\n            const bool properties_changed", "        exp7_compatible = graph_compatible;\n        if (graph_compatible) {\n            const bool properties_changed"),
            ("            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);", "            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);\n            exp7_changed = properties_changed;"),
            ("#endif // USE_CUDA_GRAPH\n\n    if (use_cuda_graph && cuda_graph_update_required)", "    exp7_warmup_after = graph->warmup_complete;\n#endif // USE_CUDA_GRAPH\n\n    if (use_cuda_graph && cuda_graph_update_required)"),
            ("    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);", "    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);\n    exp7_record(EXP7_CUDA, {cuda_ctx->device, static_cast<int64_t>(cgraph->uid),\n        exp7_enabled, exp7_compatible, exp7_changed, exp7_warmup_before, exp7_warmup_after,\n        use_cuda_graph, use_cuda_graph && cuda_graph_update_required, exp7_instance});"),
        ],
    }

    # Check all anchors before writing any patched files.
    patched = {}
    for name, replacements in edits.items():
        text = (source / name).read_text()
        prefix = suffix = ""

        if name == "src/llama-graph.cpp":
            prefix, text = text.split("// dedup helpers", 1)
            text, suffix = text.split("// impl", 1)
            prefix += "// dedup helpers"
            suffix = "// impl" + suffix

        for old, new in replacements:
            text = replace_once(text, old, new)

        text = prefix + text + suffix
        if name.endswith((".cpp", ".cu")):
            text = '#include "exp7-trace.h"\n' + text
        patched[name] = text

    for name, text in patched.items():
        (source / name).write_text(text)

    scripts = Path(__file__).resolve().parent
    shutil.copyfile(scripts / "exp7-trace.h", source / "ggml/include/exp7-trace.h")
    shutil.copyfile(scripts / "exp7-trace.cpp", source / "ggml/src/exp7-trace.cpp")
