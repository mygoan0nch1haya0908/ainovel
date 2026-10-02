"""Bounded process-local observations, never account-wide usage or prompt storage."""
from collections import deque
from threading import Lock
from time import monotonic
from uuid import uuid4


class RequestMetrics:
    def __init__(self):
        self.events = deque(maxlen=2048)
        self.active = {}
        self.lock = Lock()

    def start(self, target, operation, before, *, now=None):
        now = monotonic() if now is None else now
        with self.lock:
            previous = [e for e in self.events if e['target']==target and now-e['time']<60]
            same = [e for e in previous if operation and e['operation']==operation]
            event={'id':uuid4().hex,'target':target,'operation':operation,'time':now,
                   'chars':before.get('total_chars',0),'estimate':before.get('estimated_input_tokens',0),'usage':None}
            self.events.append(event)
            self.active[event['id']]=target
            window=previous+[event]
            growth=bool(same and event['chars']>max(2*same[0]['chars'],same[0]['chars']+1024))
            return {'metric_id':event['id'],'concurrency':sum(v==target for v in self.active.values()),
                    'requests_last_60s':len(window),'tokens_last_60s':None,
                    'known_tokens_last_60s':sum(e['usage'] or 0 for e in window),
                    'estimated_tokens_last_60s':sum(e['estimate'] for e in window),
                    'unknown_usage_requests':sum(e['usage'] is None for e in window),
                    'last_request_interval':now-previous[-1]['time'] if previous else None,
                    'operation_calls_last_60s':len(same)+1,'context_growth':growth,
                    'metrics_truncated':len(self.events)==self.events.maxlen}

    def finish(self, identifier, usage, *, now=None):
        with self.lock:
            self.active.pop(identifier,None)
            for e in reversed(self.events):
                if e['id']==identifier:
                    if isinstance(usage,dict) and all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens')):
                        e['usage']=usage['prompt_tokens']+usage['completion_tokens']
                    window=[r for r in self.events if r['target']==e['target'] and 0<=e['time']-r['time']<60]
                    unknown=sum(r['usage'] is None for r in window)
                    known=sum(r['usage'] or 0 for r in window)
                    return {'known_tokens_last_60s':known,'tokens_last_60s':None if unknown else known,
                            'unknown_usage_requests':unknown,'metrics_truncated':len(self.events)==self.events.maxlen}
            return {'metrics_truncated':True}


metrics=RequestMetrics()
