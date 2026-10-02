# Qwen adapter implementation plan

User-approved scope: integrate the confirmed Beijing Qwen endpoint with the existing offline-first novel workflow. Do not generate or approve real novel chapters during adapter testing.

## Global Constraints

- Preserve OpenAI Responses behavior and all existing approval, context, and persistence guards.
- Default real calls remain disabled; no secrets in files, logs, errors, or commits.
- Use Qwen Chat Completions with JSON Object mode and enable_thinking=false; advertise strict_structured_output=false. Include the output schema and explicit JSON instruction in the model input; existing AgentRunner validates the result against its Pydantic contract.
- Default Beijing base URL: https://dashscope.aliyuncs.com/compatible-mode/v1. Respect explicit Qwen base URL configuration. Default testing model: qwen-flash.
- Retain the 16000 context and 4000 output ceilings. No automatic retries beyond the existing bounded workflow; disable SDK retries for Qwen so cost stays bounded.
- Only synthetic small opt-in live tests; no publication, push, merge, or full novel generation in this task.

### Task 1: Qwen provider and application integration

Use test-driven development. Add a separate QwenProvider alongside OpenAIProvider, implementing capabilities, generate, diagnose. Map chat response choices[0].message.content and prompt_tokens/completion_tokens to ModelResponse. Reject truncation, refusal, missing/empty choices, malformed/non-object JSON and malformed response envelopes using safe fixed ProviderProtocolError messages. Normalize SDK authentication, timeout, connection and status failures without leaking request/response bodies. Disabled/missing-key calls must not contact the network. Diagnose configuration only and say it is not an online connectivity check.

Add Settings fields qwen_api_key (SecretStr), qwen_base_url, allow_real_qwen=false. Accept AINOVEL_QWEN_API_KEY with precedence over DASHSCOPE_API_KEY as fallback, including normal programmatic Settings construction. Do not access Windows user secrets automatically in application code; document that an existing process may require restart to inherit new environment variables.

Register a lazy qwen factory with current conservative ceilings and max_retries=0. Add qwen to the service whitelist, provider exports and both UI provider selectors. Preserve fake defaults and UI author approval behavior.

Tests: SDK wire shape via mock transport/client, system schema and input budget accounting, valid response plus AgentRunner schema failure, usage/id mapping, disabled/no-key no network, truncation/refusal/malformed/error redaction, settings alias precedence and independent opt-in, lazy default registry and workflow qwen acceptance with fake transport and approval gating. Ensure schema text appended by provider is covered by current budget calculation or implement narrowly justified accounting, not a hidden payload increase.

Document secure environment configuration, Beijing base URL override, qwen-flash selection, configuration-only diagnosis, JSON mode limitations, 4000 output-token limitation for full-length chapters, possible external charges, no model download, and live smoke-test scope. Consult existing README and test patterns rather than restructuring unrelated code.

Run focused RED then GREEN tests, run full offline suite once. Use D:/ainovel/.worktrees/phase2-orchestration-context/.venv/Scripts/python.exe with PYTHONPATH pointing at this worktree's src. No live call by implementer. Commit implementation and tests locally; report commands/results and concerns.
