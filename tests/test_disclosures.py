from pathlib import Path
import shutil
import subprocess


def test_disclosure_controls_only_toggle_their_group_and_reveal_invalid_fields():
    script = Path(__file__).resolve().parents[1] / "src/ainovel/static/disclosures.js"
    assert script.exists(), "Disclosure behavior is missing"
    node = shutil.which("node")
    assert node
    harness = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const panels = [{open:false}, {open:false}];
const unrelated = {open:false};
const listeners = {};
const buttons = ['expand','collapse'].map(action => ({
  hidden:true, dataset:{disclosureAction:action},
  addEventListener:(name,fn) => listeners[action]=fn
}));
const scope = {querySelectorAll: selector => selector === '[data-disclosure-action]' ? buttons : panels};
const globalEvents = {};
const document = {
  querySelectorAll: () => [scope],
  addEventListener: (name, fn) => globalEvents[name]=fn
};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'), {document});
assert(buttons.every(b=>!b.hidden));
listeners.expand(); assert(panels.every(p=>p.open)); assert.equal(unrelated.open,false);
listeners.collapse(); assert(panels.every(p=>!p.open));
const outer = {open:false, parentElement:null, matches:s=>s==='details'};
const inner = {open:false, parentElement:outer, matches:s=>s==='details'};
globalEvents.invalid({target:{parentElement:inner}});
assert(outer.open && inner.open);
'''
    result = subprocess.run([node, "-e", harness, str(script)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
