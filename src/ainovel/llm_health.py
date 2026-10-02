"""Opt-in diagnostic using a saved profile or AIHUBMIX_* settings; one request by default."""
import argparse
import json
import logging
import os
import random
import time
import math
from dataclasses import replace
from pathlib import Path
import sqlite3

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ainovel.models.stage import StageRoadmapVersion
from ainovel.providers.compatible import CompatibleProvider
from ainovel.providers.contracts import ModelRequest, ProviderError, ProviderTimeout
from ainovel.providers.diagnostics import FailureReason, ResponseFailure, safe_failure_code, failure_code_detail
from ainovel.providers.endpoint_policy import normalize_endpoint
from ainovel.providers.llm_response import safe_preview
from ainovel.providers.llm_diagnostic import diagnostic_for_error,retry_delay
from ainovel.services.model_profiles import ModelProfileService


class HealthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='AIHUBMIX_', env_file='.env', extra='ignore')
    api_key: SecretStr | None = None
    base_url: str = ''
    model: str = ''


def probe(provider, model, *, timeout=30.0, retries=0, synthetic_chars=0, max_output_tokens=128):
    if (type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 128
            or type(retries) is not int or not 0 <= retries <= 2
            or type(synthetic_chars) is not int or not 0 <= synthetic_chars <= 100000
            or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 180):
        raise ValueError('invalid probe limits')
    caps = provider.capabilities(model)
    output = min(max_output_tokens, caps.max_output_tokens, caps.context_window - 1)
    if output < 1:
        raise ValueError('insufficient configured context')
    payload = {'test':'synthetic'}
    if synthetic_chars:
        payload['synthetic_padding'] = ('这是诊断用的虚构测试材料，不包含真实小说或个人信息。' * (synthetic_chars // 26 + 1))[:synthetic_chars]
    request = ModelRequest(model, '只返回 {"status":"ok"}，不要解释。', payload,
        {'type':'object','properties':{'status':{'type':'string','enum':['ok']}},'required':['status'],'additionalProperties':False},
        caps.context_window - output, output, timeout, {})
    from ainovel.context import ConservativeEstimator
    from ainovel.providers.request_diagnostics import messages_for
    if ConservativeEstimator().estimate(''.join(m['content'] for m in messages_for(request))) > request.max_input_tokens:
        raise ValueError('synthetic probe exceeds configured input capacity')
    for attempt in range(1, retries + 2):
        try:
            response = provider.generate(replace(request, metadata={'attempt': str(attempt)}))
            break
        except ProviderError as error:
            diagnostic=diagnostic_for_error(error)
            if attempt > retries or diagnostic['error_category'] not in {'timeout','rate_limit'}:
                raise
            delay=retry_delay(diagnostic,attempt)
            if delay is None:
                raise
            time.sleep(delay)
    if response.structured != {'status':'ok'}:
        raise ResponseFailure(FailureReason.SCHEMA)
    return response.structured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Authorize real API calls (one by default; --retries authorizes extra attempts); may incur cost')
    parser.add_argument('--show-content', action='store_true', help='Print bounded, redacted response repr; may contain private vendor messages')
    parser.add_argument('--stage-id', help='Use the latest version of this stage, without sending its novel input')
    parser.add_argument('--database', type=Path, help='Database containing the saved stage/profile (read-only)')
    parser.add_argument('--timeout', type=float, default=30, help='Total deadline, 0 < seconds <= 180')
    parser.add_argument('--retries', type=int, choices=(0, 1, 2), default=0, help='Explicitly authorize extra timeout attempts; each may cost money')
    parser.add_argument('--synthetic-chars', type=int, default=0, help='Synthetic padding size, 0..100000; never reads novel text')
    parser.add_argument('--max-output-tokens', type=int, default=128, help='Probe output cap, 1..128; use 32 for rate-limit tests')
    parser.add_argument('--diagnostic-level', choices=('normal','debug'), default='normal', help='Debug adds safe technical metrics, never prompt text')
    args = parser.parse_args(argv)
    if not args.run:
        print('No API request made. Add --run to authorize one synthetic request; --show-content enables redacted raw diagnostics.')
        return 0
    logger = logging.getLogger('ainovel.llm')
    previous_raw=(logger.level, logger.disabled, list(logger.handlers), logger.propagate,
                  os.environ.get('AINOVEL_LLM_DEBUG_CONTENT'))
    if args.show_content:
        os.environ['AINOVEL_LLM_DEBUG_CONTENT'] = '1'
        logger = logging.getLogger('ainovel.llm')
        logger.disabled = False
        logger.setLevel(logging.DEBUG)
        logger.handlers=[logging.StreamHandler()]
        logger.propagate=False
    telemetry = logging.getLogger('ainovel.llm.telemetry')
    previous_level=telemetry.level
    previous_telemetry=(telemetry.disabled,list(telemetry.handlers),telemetry.propagate)
    telemetry.disabled = False
    telemetry.setLevel(logging.INFO if args.diagnostic_level=='debug' else logging.WARNING)
    handler = logging.StreamHandler()
    telemetry.handlers=[handler]
    telemetry.propagate=False
    try:
        if args.stage_id:
            if args.database is None or not args.database.is_file():
                raise ValueError('database required')
            path = args.database.resolve().as_posix()
            engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(f'file:{path}?mode=ro', uri=True))
            try:
                with Session(engine) as session:
                    row = session.scalars(select(StageRoadmapVersion).where(StageRoadmapVersion.stage_id == args.stage_id)
                        .order_by(StageRoadmapVersion.version_number.desc())).first()
                    if row is None or not row.model_profile_version_id:
                        raise ValueError('no saved compatible profile')
                    resolved = ModelProfileService(session).resolve_for_call(row.model_profile_version_id)
                    model, endpoint, key = resolved.model_name, resolved.endpoint, resolved.api_key
                    context, output = resolved.context_limit, resolved.output_limit
            finally:
                engine.dispose()
        else:
            settings = HealthSettings()
            if not settings.model or not settings.base_url or not settings.api_key:
                print('configuration_error: set AIHUBMIX_MODEL, AIHUBMIX_BASE_URL, AIHUBMIX_API_KEY; no request made.')
                return 2
            model, key = settings.model, settings.api_key.get_secret_value()
            endpoint = normalize_endpoint(settings.base_url, 'remote')
            context, output = 2048, 128
        provider = CompatibleProvider(endpoint, model, api_key=key, allow_real_calls=True,
            context_window_limit=context, max_output_tokens_limit=output)
        print('model=' + safe_preview(model, api_key=key))
        print('base_url=' + safe_preview(endpoint.base_url, api_key=key))
        result = probe(provider, model, timeout=args.timeout, retries=args.retries, synthetic_chars=args.synthetic_chars, max_output_tokens=args.max_output_tokens)
        print('response_status=ok parsed_json=' + json.dumps(result))
        return 0
    except ProviderError as error:
        code = safe_failure_code(error)
        print('response_status=failed classification=' + code)
        print(failure_code_detail(code))
        diagnostic=diagnostic_for_error(error)
        if args.diagnostic_level=='normal':
            diagnostic={k:diagnostic[k] for k in ('error_category','error_subtype','error_layer','confidence','evidence','recommended_actions','retry_after')}
        print(json.dumps(diagnostic,ensure_ascii=False))
        return 1
    except Exception:
        # Configuration/validation exceptions may embed secrets; never print them.
        print('configuration_error: check model/base_url/profile/key availability; details suppressed for privacy.')
        return 2
    finally:
        telemetry.disabled,telemetry.handlers,telemetry.propagate=previous_telemetry
        telemetry.setLevel(previous_level)
        logger.setLevel(previous_raw[0])
        logger.disabled,logger.handlers,logger.propagate=previous_raw[1:4]
        if previous_raw[4] is None:
            os.environ.pop('AINOVEL_LLM_DEBUG_CONTENT',None)
        else:
            os.environ['AINOVEL_LLM_DEBUG_CONTENT']=previous_raw[4]


if __name__ == '__main__':
    raise SystemExit(main())
