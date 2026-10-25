#pragma once

#include "chat.h"
#include "server-http.h"
#include "server-task.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

struct tn_runtime_handle;
struct tn_turn_handle;

enum class server_tn_phase : uint32_t {
    governance_read = 1,
    commitment_echo = 2,
    section_selection = 3,
    governed_action   = 4,
};

struct server_tn_turn {
    explicit server_tn_turn(tn_turn_handle * handle, server_tn_phase phase);
    ~server_tn_turn();

    server_tn_turn(const server_tn_turn &) = delete;
    server_tn_turn & operator=(const server_tn_turn &) = delete;

    tn_turn_handle * handle = nullptr;
    server_tn_phase phase;
};

struct server_tn_model_result {
    int64_t task_id;
    int32_t slot_id;
    bool truncated;
    bool stop_limit;
    std::string raw_model_output;
    common_chat_msg parsed;
};

using server_tn_run_turn = std::function<server_tn_model_result(
        const std::string & rust_request_json,
        const std::shared_ptr<server_tn_turn> & turn,
        const std::function<bool()> & should_stop)>;

class server_tn_runtime {
public:
    explicit server_tn_runtime(tn_runtime_handle * handle);
    ~server_tn_runtime();

    server_tn_runtime(const server_tn_runtime &) = delete;
    server_tn_runtime & operator=(const server_tn_runtime &) = delete;

private:
    friend server_http_res_ptr server_tn_handle(
            const server_http_req &, server_tn_runtime &, const server_tn_run_turn &);

    tn_runtime_handle * handle_ = nullptr;
};

std::unique_ptr<server_tn_runtime> server_tn_runtime_create(
        const std::string & config_json,
        const std::string & model_path);

server_http_res_ptr server_tn_handle(
        const server_http_req & req,
        server_tn_runtime & runtime,
        const server_tn_run_turn & run_turn);

std::string server_tn_runnable_manifest(const server_task & task, bool effective_ctx_shift);

bool server_tn_prepare_model_turn(
        server_tn_turn & turn,
        const std::string & request_json,
        const std::string & rendered_prompt,
        const std::string & runnable_manifest,
        std::string & error_json);

bool server_tn_begin_model_turn(
        server_tn_turn & turn,
        int64_t task_id,
        int32_t slot_id,
        const std::string & request_json,
        const std::string & rendered_prompt,
        const std::string & runnable_manifest,
        std::string & error_json);

// This is the exact helper invoked immediately before a governed task is
// installed on a slot. It rebuilds the manifest from `task`; callers may only
// install the task when it returns true.
bool server_tn_authorize_task_for_slot(
        const server_task & task,
        const std::shared_ptr<server_tn_turn> & turn,
        const std::string & prepared_manifest,
        const std::string & request_json,
        const std::string & rendered_prompt,
        int32_t slot_id,
        bool effective_ctx_shift,
        std::string & error_json);
