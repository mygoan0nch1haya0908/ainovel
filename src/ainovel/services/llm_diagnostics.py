"""Persist in the existing audit stream; cooldown survives process restarts."""
from datetime import datetime, timezone
from sqlalchemy import select
from ainovel.models.audit import AuditEvent
from ainovel.providers.llm_diagnostic import diagnostic_for_error, public_diagnostic, retry_delay, number


def failure_details(error, *, now=None, **identity):
    diagnostic=diagnostic_for_error(error)
    delay=0
    if diagnostic['error_category']=='rate_limit' or diagnostic.get('retry_after') is not None:
        delay=diagnostic.get('retry_after')
        if delay is None:
            delay=retry_delay({},identity.get('attempt_number',1))
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None: now=now.replace(tzinfo=timezone.utc)
    return {**identity,'diagnostic':diagnostic,'cooldown_seconds':delay,
            'retry_not_before':now.timestamp()+delay if delay else None}


def enforce_cooldown(session,entity_id,now=None):
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None: now=now.replace(tzinfo=timezone.utc)
    events=session.scalars(select(AuditEvent).where(AuditEvent.entity_id==entity_id,
        AuditEvent.action=='llm_call_failed').order_by(AuditEvent.created_at.desc()).limit(10)).all()
    for event in events:
        delay=number(event.details.get('cooldown_seconds'))
        if not delay: continue
        deadline=event.details.get('retry_not_before')
        if type(deadline) in (int,float) and 0<deadline<10_000_000_000:
            if now.timestamp()<deadline: raise ValueError('model retry cooldown is active')
            continue
        created=event.created_at
        if created.tzinfo is None: created=created.replace(tzinfo=timezone.utc)
        if delay is not None and (now-created).total_seconds()<delay:
            raise ValueError('model retry cooldown is active')


def diagnostic_rows(session,entity_id):
    events=session.scalars(select(AuditEvent).where(AuditEvent.entity_id==entity_id,
        AuditEvent.action=='llm_call_failed').order_by(AuditEvent.created_at)).all()
    return [{**e.details,'diagnostic':public_diagnostic(e.details.get('diagnostic'))} for e in events if isinstance(e.details,dict)]
