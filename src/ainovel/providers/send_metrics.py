"""Process-local send attempts, not provider acceptance or account-wide billing."""
from collections import deque, OrderedDict
from threading import Lock
from time import monotonic
from uuid import uuid4
from ainovel.providers.contracts import ProviderUnavailable


class CallBudgetExceeded(ProviderUnavailable):
    pass


class SendMetrics:
    def __init__(self):
        self.events = deque(maxlen=4096)
        self.active = {}
        self.operations = OrderedDict()
        self.lock = Lock()

    def start(self, target, operation, estimate, *, now=None, budget=128):
        now = monotonic() if now is None else now
        with self.lock:
            count = self.operations.get(operation, 0) if operation else 0
            if operation and count >= budget:
                raise CallBudgetExceeded('local LLM call budget exceeded')
            if operation:
                self.operations[operation] = count + 1
                self.operations.move_to_end(operation)
                # Refuse new operations at capacity rather than evict active budgets.
                if len(self.operations) > 4096:
                    self.operations.pop(operation)
                    raise CallBudgetExceeded('local LLM call budget registry full')
            identifier = uuid4().hex
            event = dict(id=identifier, target=target, operation=operation, time=now,
                         estimate=estimate, usage=None)
            self.active[identifier] = target
            event['peak'] = len(self.active)
            self.events.append(event)
            d = {'llm_call_id':identifier, 'user_request_actual_calls':count+1,
                 'actual_process_concurrency':len(self.active),
                 'actual_model_concurrency':sum(t==target for t in self.active.values()),
                 'actual_peak_concurrency_last_60s':max(e['peak'] for e in self.events if now-e['time']<60),
                 'send_metrics_truncated':len(self.events)==self.events.maxlen}
            for seconds in (10, 60, 300):
                window = [e for e in self.events if 0<=now-e['time']<seconds]
                d[f'actual_process_requests_last_{seconds}s'] = len(window)
                d[f'actual_operation_requests_last_{seconds}s'] = sum(e['operation']==operation for e in window) if operation else None
                same = [e for e in window if e['target']==target]
                d[f'actual_requests_last_{seconds}s'] = len(same)
                if seconds!=10:
                    d[f'actual_known_tokens_last_{seconds}s'] = sum(e['usage'] or 0 for e in same)
                    d[f'actual_tokens_last_{seconds}s'] = None if any(e['usage'] is None for e in same) else sum(e['usage'] for e in same)
                    d[f'actual_estimated_tokens_last_{seconds}s'] = sum(e['estimate'] for e in same)
            return d

    def finish(self, identifier, usage):
        with self.lock:
            self.active.pop(identifier, None)
            for e in reversed(self.events):
                if e['id']==identifier:
                    if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                        e['usage'] = usage['prompt_tokens']+usage['completion_tokens']
                    result={}
                    for seconds in (60,300):
                        window=[r for r in self.events if r['target']==e['target'] and 0<=e['time']-r['time']<seconds]
                        result[f'actual_known_tokens_last_{seconds}s']=sum(r['usage'] or 0 for r in window)
                        result[f'actual_tokens_last_{seconds}s']=None if any(r['usage'] is None for r in window) else sum(r['usage'] for r in window)
                    return result
            return {}

    def close_operation(self, operation):
        with self.lock:
            self.operations.pop(operation, None)
