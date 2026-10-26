#ifndef TN_RUNTIME_H
#define TN_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tn_runtime_handle tn_runtime_handle;
typedef struct tn_session_handle tn_session_handle;
typedef struct tn_turn_handle tn_turn_handle;

typedef struct tn_bytes_view {
    const uint8_t *ptr;
    size_t len;
} tn_bytes_view;

typedef struct tn_owned_bytes {
    uint8_t *ptr;
    size_t len;
} tn_owned_bytes;

typedef enum tn_status {
    TN_STATUS_OK = 0,
    TN_STATUS_ERROR = 1,
    TN_STATUS_PANIC = 2
} tn_status;

typedef enum tn_turn_phase {
    TN_TURN_GOVERNANCE_READ = 1,
    TN_TURN_COMMITMENT_ECHO = 2,
    TN_TURN_SECTION_SELECTION = 3,
    TN_TURN_GOVERNED_ACTION = 4
} tn_turn_phase;

typedef enum tn_action_outcome {
    TN_ACTION_CONTINUE = 1,
    TN_ACTION_RELEASE = 2
} tn_action_outcome;

typedef struct tn_native_tool_call {
    tn_bytes_view id;
    tn_bytes_view name;
    tn_bytes_view normalized_arguments;
} tn_native_tool_call;

typedef struct tn_native_tool_result {
    size_t call_count;
    const tn_native_tool_call *call;
    tn_bytes_view content;
    tn_bytes_view raw_model_output;
    uint8_t truncated;
    uint8_t stop_limit;
} tn_native_tool_result;

/* Check the observed model SHA-256 against receiver configuration. */
int32_t tn_governance_model_binding_authorized(
                          tn_bytes_view observed_model_sha256,
                          uint8_t *out_authorized,
                          tn_owned_bytes *out_error);
int32_t tn_governance_model_file_authorized(
                          tn_bytes_view config_json,
                          tn_bytes_view model_path,
                          uint8_t *out_authorized,
                          tn_owned_bytes *out_error);
int32_t tn_runtime_create(tn_bytes_view config_json,
                          tn_runtime_handle **out_runtime,
                          tn_owned_bytes *out_error);
int32_t tn_session_create(tn_runtime_handle *runtime,
                          tn_bytes_view request_json,
                          tn_session_handle **out_session,
                          tn_owned_bytes *out_error);
int32_t tn_session_next_model_turn(tn_session_handle *session,
                          tn_turn_handle **out_turn,
                          uint32_t *out_phase,
                          tn_owned_bytes *out_request_json,
                          tn_owned_bytes *out_error);
int32_t tn_turn_prepare(tn_turn_handle *turn,
                          tn_bytes_view request_json,
                          tn_bytes_view rendered_prompt,
                          tn_bytes_view runnable_manifest,
                          tn_owned_bytes *out_error);
int32_t tn_turn_begin(tn_turn_handle *turn,
                          int64_t task_id,
                          int32_t slot_id,
                          tn_bytes_view request_json,
                          tn_bytes_view rendered_prompt,
                          tn_bytes_view runnable_manifest,
                          tn_owned_bytes *out_error);
int32_t tn_session_accept_model_action(tn_session_handle *session,
                          tn_turn_handle *turn,
                          const tn_native_tool_result *result,
                          uint32_t *out_outcome,
                          tn_owned_bytes *out_release_json,
                          tn_owned_bytes *out_error);
int32_t tn_session_abort(tn_session_handle *session,
                          tn_owned_bytes *out_error);

void tn_bytes_free(tn_owned_bytes *value);
void tn_turn_free(tn_turn_handle *turn);
void tn_session_free(tn_session_handle *session);
void tn_runtime_free(tn_runtime_handle *runtime);

#ifdef __cplusplus
}
#endif
#endif
