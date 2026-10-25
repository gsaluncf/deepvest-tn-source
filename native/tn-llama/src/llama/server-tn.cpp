#include "server-tn.h"

#include "speculative.h"
#include "tn_runtime.h"

#include <cctype>
#include <cinttypes>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

tn_bytes_view bytes_view(const std::string & value) {
    return {
        reinterpret_cast<const uint8_t *>(value.data()),
        value.size(),
    };
}

std::string take_owned(tn_owned_bytes & value) {
    std::string result;
    if (value.ptr != nullptr && value.len != 0) {
        result.assign(reinterpret_cast<const char *>(value.ptr), value.len);
    }
    tn_bytes_free(&value);
    return result;
}

std::string phase_name(server_tn_phase phase) {
    switch (phase) {
        case server_tn_phase::governance_read:   return "governance_read";
        case server_tn_phase::commitment_echo:   return "commitment_echo";
        case server_tn_phase::section_selection: return "section_selection";
        case server_tn_phase::governed_action:   return "governed_action";
    }
    throw std::runtime_error("invalid TN turn phase");
}

server_tn_phase checked_phase(uint32_t phase) {
    switch (phase) {
        case TN_TURN_GOVERNANCE_READ:   return server_tn_phase::governance_read;
        case TN_TURN_COMMITMENT_ECHO:   return server_tn_phase::commitment_echo;
        case TN_TURN_SECTION_SELECTION: return server_tn_phase::section_selection;
        case TN_TURN_GOVERNED_ACTION:   return server_tn_phase::governed_action;
        default: throw std::runtime_error("Rust returned an invalid TN turn phase");
    }
}

std::string task_type_name(server_task_type type) {
    switch (type) {
        case SERVER_TASK_TYPE_COMPLETION:   return "completion";
        case SERVER_TASK_TYPE_EMBEDDING:    return "embedding";
        case SERVER_TASK_TYPE_RERANK:       return "rerank";
        case SERVER_TASK_TYPE_INFILL:       return "infill";
        case SERVER_TASK_TYPE_CANCEL:       return "cancel";
        case SERVER_TASK_TYPE_CONTROL:      return "control";
        case SERVER_TASK_TYPE_NEXT_RESPONSE:return "next_response";
        case SERVER_TASK_TYPE_METRICS:      return "metrics";
        case SERVER_TASK_TYPE_SLOT_GET:     return "slot_get";
        case SERVER_TASK_TYPE_SLOT_SAVE:    return "slot_save";
        case SERVER_TASK_TYPE_SLOT_RESTORE: return "slot_restore";
        case SERVER_TASK_TYPE_SLOT_ERASE:   return "slot_erase";
        case SERVER_TASK_TYPE_GET_LORA:     return "get_lora";
        case SERVER_TASK_TYPE_SET_LORA:     return "set_lora";
    }
    throw std::runtime_error("invalid llama task type");
}

std::string response_type_name(task_response_type type) {
    switch (type) {
        case TASK_RESPONSE_TYPE_NONE:      return "none";
        case TASK_RESPONSE_TYPE_OAI_CHAT:  return "oai_chat";
        case TASK_RESPONSE_TYPE_OAI_CMPL:  return "oai_completion";
        case TASK_RESPONSE_TYPE_OAI_RESP:  return "oai_response";
        case TASK_RESPONSE_TYPE_OAI_ASR:   return "oai_transcription";
        case TASK_RESPONSE_TYPE_OAI_EMBD:  return "oai_embedding";
        case TASK_RESPONSE_TYPE_ANTHROPIC: return "anthropic";
    }
    throw std::runtime_error("invalid llama response type");
}

std::string grammar_type_name(common_grammar_type type) {
    switch (type) {
        case COMMON_GRAMMAR_TYPE_NONE:          return "none";
        case COMMON_GRAMMAR_TYPE_USER:          return "user";
        case COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT: return "output_format";
        case COMMON_GRAMMAR_TYPE_TOOL_CALLS:    return "tool_calls";
    }
    throw std::runtime_error("invalid llama grammar type");
}

json lora_projection(const std::map<int, float> & lora) {
    json result = json::array();
    for (const auto & entry : lora) {
        result.push_back({
            {"id", entry.first},
            {"scale", entry.second},
        });
    }
    return result;
}

json speculative_projection(const common_params_speculative & speculative) {
    json types = json::array();
    for (const common_speculative_type type : speculative.types) {
        types.push_back(common_speculative_type_to_str(type));
    }
    return {{"types", std::move(types)}};
}

json token_projection(const server_tokens & tokens) {
    json result = json::array();
    if (tokens.has_mtmd) {
        return result;
    }
    for (const llama_token token : tokens.get_tokens()) {
        result.push_back(token);
    }
    return result;
}

json token_sequences_projection(const std::vector<llama_tokens> & sequences) {
    json result = json::array();
    for (const llama_tokens & sequence : sequences) {
        json projected = json::array();
        for (const llama_token token : sequence) {
            projected.push_back(token);
        }
        result.push_back(std::move(projected));
    }
    return result;
}

server_http_res_ptr safe_error_response(const std::string & error_json, int status) {
    std::string code = "tn_governed_request_denied";
    try {
        const json parsed = json::parse(error_json);
        if (parsed.contains("code") && parsed.at("code").is_string()) {
            code = parsed.at("code").get<std::string>();
        }
    } catch (...) {
        // The Rust error is for the local log only if its adapter shape is lost.
    }
    auto response = std::make_unique<server_http_res>();
    response->status = status;
    response->data = json({
        {"error", {
            {"message", "TN governed request denied"},
            {"type", "tn_governance_error"},
            {"code", code},
        }},
    }).dump();
    const std::string http_event = json({
        {"event", "http_returned"},
        {"status", status},
        {"error_code", code},
        {"body_bytes", response->data.size()},
    }).dump();
    SRV_INF("TN_EVENT %s\n", http_event.c_str());
    return response;
}

class session_guard {
public:
    explicit session_guard(tn_session_handle * handle) : handle_(handle) {}

    ~session_guard() {
        if (handle_ == nullptr) {
            return;
        }
        if (!released_) {
            tn_owned_bytes error{};
            (void) tn_session_abort(handle_, &error);
            tn_bytes_free(&error);
        }
        tn_session_free(handle_);
    }

    session_guard(const session_guard &) = delete;
    session_guard & operator=(const session_guard &) = delete;

    tn_session_handle * get() const { return handle_; }
    void mark_released() { released_ = true; }

private:
    tn_session_handle * handle_ = nullptr;
    bool released_ = false;
};

bool is_multipart(const server_http_req & request) {
    for (const auto & header : request.headers) {
        std::string name = header.first;
        for (char & value : name) {
            value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
        }
        if (name != "content-type") {
            continue;
        }
        std::string value = header.second;
        for (char & ch : value) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return value.rfind("multipart/", 0) == 0;
    }
    return false;
}

} // namespace

server_tn_turn::server_tn_turn(tn_turn_handle * handle, server_tn_phase phase)
        : handle(handle), phase(phase) {
    if (handle == nullptr) {
        throw std::invalid_argument("TN turn handle is null");
    }
}

server_tn_turn::~server_tn_turn() {
    tn_turn_free(handle);
}

server_tn_runtime::server_tn_runtime(tn_runtime_handle * handle) : handle_(handle) {
    if (handle == nullptr) {
        throw std::invalid_argument("TN runtime handle is null");
    }
}

server_tn_runtime::~server_tn_runtime() {
    tn_runtime_free(handle_);
}

std::unique_ptr<server_tn_runtime> server_tn_runtime_create(
        const std::string & config_json,
        const std::string & model_path) {
    uint8_t model_authorized = 0;
    tn_owned_bytes model_error{};
    const int32_t model_status = tn_governance_model_file_authorized(
            bytes_view(config_json), bytes_view(model_path), &model_authorized, &model_error);
    const std::string model_error_json = take_owned(model_error);
    if (model_status != TN_STATUS_OK) {
        throw std::runtime_error("TN model hash failed: " + model_error_json);
    }
    if (model_authorized != 1) {
        throw std::runtime_error("TN model hash does not match receiver policy");
    }

    tn_runtime_handle * handle = nullptr;
    tn_owned_bytes error{};
    const int32_t status = tn_runtime_create(bytes_view(config_json), &handle, &error);
    const std::string error_json = take_owned(error);
    if (status != TN_STATUS_OK) {
        throw std::runtime_error("TN runtime creation failed: " + error_json);
    }
    return std::make_unique<server_tn_runtime>(handle);
}

std::string server_tn_runnable_manifest(const server_task & task, bool effective_ctx_shift) {
    const task_params & params = task.params;
    const common_params_sampling & sampling = params.sampling;
    const common_chat_parser_params & parser = params.chat_parser_params;

    const json manifest = {
        {"schema", "tn.llama.runnable-task.v1"},
        {"effective_ctx_shift", effective_ctx_shift},
        {"task", {
            {"type", task_type_name(task.type)},
            {"tokens", token_projection(task.tokens)},
            {"tokens_has_mtmd", task.tokens.has_mtmd},
            {"id_slot", task.id_slot},
            {"index", task.index},
            {"id_parent", task.id_parent},
            {"child_count", task.child_tasks.size()},
            {"cli", task.cli},
            {"cli_prompt_byte_length", task.cli_prompt.size()},
            {"cli_file_count", task.cli_files.size()},
        }},
        {"params", {
            {"stream", params.stream},
            {"cache_prompt", params.cache_prompt},
            {"n_cache_reuse", params.n_cache_reuse},
            {"n_cmpl", params.n_cmpl},
            {"n_predict", params.n_predict},
            {"n_keep", params.n_keep},
            {"n_discard", params.n_discard},
            {"n_indent", params.n_indent},
            {"t_max_predict_ms", params.t_max_predict_ms},
            {"lora", lora_projection(params.lora)},
            {"stop", params.antiprompt},
            {"response_fields", params.response_fields},
            {"post_sampling_probs", params.post_sampling_probs},
            {"res_type", response_type_name(params.res_type)},
            {"upstream_params_json", params.to_json(false)},
        }},
        {"sampling_native", {
            {"n_prev", sampling.n_prev},
            {"logit_bias_eog", params.format_logit_bias(sampling.logit_bias_eog)},
            {"grammar_type", grammar_type_name(sampling.grammar.type)},
            {"sampling_generation_prompt", sampling.generation_prompt},
            {"reasoning_budget_tokens", sampling.reasoning_budget_tokens},
            {"reasoning_budget_start", sampling.reasoning_budget_start},
            {"reasoning_budget_end", token_sequences_projection(sampling.reasoning_budget_end)},
            {"reasoning_budget_forced", sampling.reasoning_budget_forced},
            {"reasoning_budget_message", sampling.reasoning_budget_message},
            {"reasoning_control", sampling.reasoning_control},
            {"backend_sampling", sampling.backend_sampling},
        }},
        {"speculative", speculative_projection(params.speculative)},
        {"chat_parser", {
            {"format", common_chat_format_name(parser.format)},
            {"reasoning_format", common_reasoning_format_name(parser.reasoning_format)},
            {"reasoning_in_content", parser.reasoning_in_content},
            {"generation_prompt", parser.generation_prompt},
            {"parse_tool_calls", parser.parse_tool_calls},
            {"is_continuation", parser.is_continuation},
            {"echo", parser.echo},
            {"debug", parser.debug},
        }},
    };
    return manifest.dump();
}

bool server_tn_prepare_model_turn(
        server_tn_turn & turn,
        const std::string & request_json,
        const std::string & rendered_prompt,
        const std::string & runnable_manifest,
        std::string & error_json) {
    tn_owned_bytes error{};
    const int32_t status = tn_turn_prepare(
            turn.handle,
            bytes_view(request_json),
            bytes_view(rendered_prompt),
            bytes_view(runnable_manifest),
            &error);
    error_json = take_owned(error);
    if (status != TN_STATUS_OK && error_json.empty()) {
        error_json = std::string(R"({"code":"tn_ffi_error","stage":"ffi","status":)")
                + std::to_string(status) + "}";
    }
    return status == TN_STATUS_OK;
}

bool server_tn_begin_model_turn(
        server_tn_turn & turn,
        int64_t task_id,
        int32_t slot_id,
        const std::string & request_json,
        const std::string & rendered_prompt,
        const std::string & runnable_manifest,
        std::string & error_json) {
    tn_owned_bytes error{};
    const int32_t status = tn_turn_begin(
            turn.handle,
            task_id,
            slot_id,
            bytes_view(request_json),
            bytes_view(rendered_prompt),
            bytes_view(runnable_manifest),
            &error);
    error_json = take_owned(error);
    if (status != TN_STATUS_OK && error_json.empty()) {
        error_json = std::string(R"({"code":"tn_ffi_error","stage":"ffi","status":)")
                + std::to_string(status) + "}";
    }
    return status == TN_STATUS_OK;
}

bool server_tn_authorize_task_for_slot(
        const server_task & task,
        const std::shared_ptr<server_tn_turn> & turn,
        const std::string & prepared_manifest,
        const std::string & request_json,
        const std::string & rendered_prompt,
        int32_t slot_id,
        bool effective_ctx_shift,
        std::string & error_json) {
    if (task.type != SERVER_TASK_TYPE_COMPLETION) {
        error_json = R"({"code":"tn_task_type_denied","stage":"llama"})";
        return false;
    }
    if (!turn) {
        error_json = R"({"code":"tn_turn_missing","stage":"llama"})";
        return false;
    }
    if (prepared_manifest.empty()) {
        error_json = R"({"code":"tn_turn_not_prepared","stage":"llama"})";
        return false;
    }
    const std::string actual_manifest = server_tn_runnable_manifest(task, effective_ctx_shift);
    return server_tn_begin_model_turn(
            *turn,
            task.id,
            slot_id,
            request_json,
            rendered_prompt,
            actual_manifest,
            error_json);
}

server_http_res_ptr server_tn_handle(
        const server_http_req & req,
        server_tn_runtime & runtime,
        const server_tn_run_turn & run_turn) {
    const std::string route_event = json({
        {"event", "route_received"},
        {"path", req.path},
        {"body_bytes", req.body.size()},
    }).dump();
    SRV_INF("TN_EVENT %s\n", route_event.c_str());

    if (is_multipart(req) || !req.files.empty()) {
        return safe_error_response(
                R"({"code":"tn_multipart_denied","stage":"ffi"})", 400);
    }

    tn_session_handle * raw_session = nullptr;
    tn_owned_bytes error{};
    int32_t status = tn_session_create(
            runtime.handle_, bytes_view(req.body), &raw_session, &error);
    std::string error_json = take_owned(error);
    if (status != TN_STATUS_OK) {
        SRV_WRN("TN_EVT session_denied error=%s\n", error_json.c_str());
        return safe_error_response(error_json, 400);
    }
    session_guard session(raw_session);

    for (;;) {
        tn_turn_handle * raw_turn = nullptr;
        uint32_t raw_phase = 0;
        tn_owned_bytes request_body{};
        status = tn_session_next_model_turn(
                session.get(), &raw_turn, &raw_phase, &request_body, &error);
        error_json = take_owned(error);
        if (status != TN_STATUS_OK) {
            SRV_WRN("TN_EVT turn_issue_denied error=%s\n", error_json.c_str());
            return safe_error_response(error_json, 400);
        }

        std::shared_ptr<server_tn_turn> turn;
        server_tn_phase actual_phase;
        try {
            actual_phase = checked_phase(raw_phase);
            turn = std::make_shared<server_tn_turn>(raw_turn, actual_phase);
        } catch (...) {
            tn_turn_free(raw_turn);
            throw;
        }
        const std::string rust_request_json = take_owned(request_body);
        SRV_INF("TN_EVT turn_issued phase=%s body_bytes=%zu\n",
                phase_name(actual_phase).c_str(), rust_request_json.size());

        server_tn_model_result result;
        try {
            result = run_turn(rust_request_json, turn, req.should_stop);
        } catch (const std::exception & model_error) {
            SRV_ERR("TN model turn failed in phase %s: %s\n",
                    phase_name(actual_phase).c_str(), model_error.what());
            return safe_error_response(
                    R"({"code":"tn_model_turn_failed","stage":"llama"})", 500);
        }
        json parsed_tool_names = json::array();
        json parsed_tool_argument_bytes = json::array();
        for (const auto & parsed_call : result.parsed.tool_calls) {
            parsed_tool_names.push_back(parsed_call.name);
            parsed_tool_argument_bytes.push_back(parsed_call.arguments.size());
        }
        const std::string result_event = json({
            {"event", "native_model_result"},
            {"phase", phase_name(actual_phase)},
            {"task_id", result.task_id},
            {"slot_id", result.slot_id},
            {"tool_call_count", result.parsed.tool_calls.size()},
            {"tool_names", parsed_tool_names},
            {"tool_argument_bytes", parsed_tool_argument_bytes},
            {"raw_output_bytes", result.raw_model_output.size()},
            {"parsed_content_bytes", result.parsed.content.size()},
            {"truncated", result.truncated},
            {"stop_limit", result.stop_limit},
            {"incomplete", result.truncated || result.stop_limit},
        }).dump();
        SRV_INF("TN_EVENT %s\n", result_event.c_str());

        tn_native_tool_call native_call{};
        const tn_native_tool_call * call_pointer = nullptr;
        if (result.parsed.tool_calls.size() == 1) {
            const common_chat_tool_call & call = result.parsed.tool_calls.front();
            native_call = {
                bytes_view(call.id),
                bytes_view(call.name),
                bytes_view(call.arguments),
            };
            call_pointer = &native_call;
        }
        const tn_native_tool_result native_result = {
            result.parsed.tool_calls.size(),
            call_pointer,
            bytes_view(result.parsed.content),
            bytes_view(result.raw_model_output),
            static_cast<uint8_t>(result.truncated),
            static_cast<uint8_t>(result.stop_limit),
        };
        uint32_t outcome = 0;
        tn_owned_bytes release{};
        status = tn_session_accept_model_action(
                session.get(), turn->handle, &native_result, &outcome, &release, &error);
        error_json = take_owned(error);
        if (status != TN_STATUS_OK) {
            tn_bytes_free(&release);
            SRV_WRN("TN_EVT native_result_denied phase=%s task_id=%" PRId64 " error=%s\n",
                    phase_name(actual_phase).c_str(), result.task_id, error_json.c_str());
            return safe_error_response(error_json, 400);
        }
        if (outcome == TN_ACTION_CONTINUE) {
            if (release.ptr != nullptr || release.len != 0) {
                tn_bytes_free(&release);
                return safe_error_response(
                        R"({"code":"tn_action_outcome_invalid","stage":"llama"})", 500);
            }
            continue;
        }
        if (outcome != TN_ACTION_RELEASE || release.ptr == nullptr || release.len == 0) {
            tn_bytes_free(&release);
            return safe_error_response(
                    R"({"code":"tn_action_outcome_invalid","stage":"llama"})", 500);
        }

        auto response = std::make_unique<server_http_res>();
        response->status = 200;
        response->data = take_owned(release);
        session.mark_released();
        const std::string http_event = json({
            {"event", "http_returned"},
            {"status", 200},
            {"error_code", nullptr},
            {"task_id", result.task_id},
            {"body_bytes", response->data.size()},
        }).dump();
        SRV_INF("TN_EVENT %s\n", http_event.c_str());
        return response;
    }
}
