"""Opt-in serial synthetic probes. Stop at the first failure, never load novel input."""
import argparse
import time
from datetime import datetime, timezone
from ainovel.llm_health import main as health_main


def run_suite(probe, *, sleep=time.sleep):
    def wait(seconds):
        while seconds:
            part=min(seconds,30)
            sleep(part)
            seconds-=part
    plan=[('minimum',70,0)]+[(f'low_frequency_{i}',15,0) for i in range(1,6)]+[
        ('size_A',70,0),('size_B',70,10000)]
    for label, delay, size in plan:
        print(f'probe={label} waiting_seconds={delay}',flush=True)
        wait(delay)
        print(f'probe={label} timestamp={datetime.now(timezone.utc).isoformat()}',flush=True)
        result=probe(size)
        print(f'probe={label} exit_code={result}',flush=True)
        if result:
            print('Stopped after failure. Remaining tests UNVERIFIED; no automatic retries.',flush=True)
            return result
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Authorize at most eight serial synthetic API requests; may incur cost')
    parser.add_argument('--stage-id',required=True)
    parser.add_argument('--database',required=True)
    args=parser.parse_args(argv)
    if not args.run:
        print('No API request made. Add --run to authorize the bounded suite.')
        return 0
    def probe(size):
        return health_main(['--run','--stage-id',args.stage_id,'--database',args.database,
            '--timeout','120','--retries','0','--max-output-tokens','32',
            '--synthetic-chars',str(size),'--diagnostic-level','debug','--show-content'])
    return run_suite(probe)


if __name__=='__main__':
    raise SystemExit(main())
