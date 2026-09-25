from __future__ import annotations
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import pytest
from qichi.dialogue.model_capability import ModelCapability,ProviderCapabilityEvidence
from qichi.readiness import *
from qichi.readiness import _pid_alive_posix, _pid_alive_windows
from qichi.storage.database import Database
from qichi.storage.migrations import SCHEMA_VERSION
from test_interactions import NOW,OWNER,BOT

def capability(tokens=262144):return ModelCapability("m",tokens,"p",ProviderCapabilityEvidence("p","m",tokens,"x",NOW))
BUILD_ID=compute_build_id(Path(__file__).parents[1])
def marker(lock,**overrides):
    value={"schema":"qichi-ready","version":READY_MARKER_VERSION,"instance_id":lock.instance_id,"lock_identity":lock.lock_identity,"pid":lock.pid,"parent_pid":lock.parent_pid,"owner_qq":OWNER,"bot_qq":BOT,"database_schema_version":SCHEMA_VERSION,"last_recovered_sequence":-1,"ws_connection_id":"ws","memory_worker_id":"mem","initiative_worker_id":"init","started_at_utc":NOW.isoformat(),"provider":"p","model":"m","context_window":262144,"build_id":BUILD_ID}
    value.update(overrides);return ReadyMarker.from_dict(value)
def legacy_marker(lock,version=2):
    """Raw marker text from before build evidence existed; never carries build_id."""
    value={"schema":"qichi-ready","version":version,"instance_id":lock.instance_id,"lock_identity":lock.lock_identity,"pid":lock.pid,"parent_pid":lock.parent_pid,"owner_qq":OWNER,"bot_qq":BOT,"database_schema_version":SCHEMA_VERSION,"last_recovered_sequence":-1,"ws_connection_id":"ws","memory_worker_id":"mem","initiative_worker_id":"init","started_at_utc":NOW.isoformat()}
    if version>=2:value.update({"provider":"p","model":"m","context_window":262144})
    return value
def runtime_tree(root):
    (root/"src"/"qichi").mkdir(parents=True);(root/"migrations").mkdir()
    (root/"src"/"qichi"/"app.py").write_text("x=1\n",encoding="utf-8")
    (root/"migrations"/"0001_initial.sql").write_text("CREATE TABLE t(x);\n",encoding="utf-8")

def test_lock_marker_checker_exact_reverse_binding_and_foreign_cleanup(tmp_path):
    lp=tmp_path/"lock";mp=tmp_path/"ready";lock=InstanceLock(lp,instance_id="i",lock_identity="l",clock=lambda:NOW);record=lock.acquire(pid=os.getpid(),parent_pid=os.getpid())
    write_ready_marker(mp,marker(record),lock_path=lp);assert check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,pid_probe=lambda p:True).build_id==BUILD_ID
    raw=json.loads(lp.read_text());raw["instance_id"]="foreign";lp.write_text(json.dumps(raw));assert lock.release() is False and lp.exists()

def test_checker_rejects_corrupt_foreign_and_dead(tmp_path):
    lp=tmp_path/"lock";mp=tmp_path/"ready";record=InstanceLock(lp,clock=lambda:NOW).acquire(pid=10,parent_pid=11);write_ready_marker(mp,marker(record),lock_path=lp)
    with pytest.raises(ReadinessError):check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,pid_probe=lambda p:False)
    raw=json.loads(mp.read_text());raw["extra"]=1;mp.write_text(json.dumps(raw))
    with pytest.raises(ReadinessError):check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,pid_probe=lambda p:True)

def test_atomic_write_failure_leaves_no_marker(tmp_path,monkeypatch):
    lp=tmp_path/"lock";mp=tmp_path/"ready";record=InstanceLock(lp,clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    monkeypatch.setattr(os,"replace",lambda *_:(_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):write_ready_marker(mp,marker(record),lock_path=lp)
    assert not mp.exists()

class Kernel:
    class Function:
        def __init__(self,call):self.call=call
        def __call__(self,*args):return self.call(*args)
    def __init__(self,handle,active=True):
        self.handle=handle;self.active=active;self.closed=[]
        self.OpenProcess=self.Function(lambda *_:self.handle)
        self.GetExitCodeProcess=self.Function(self._exit)
        self.CloseHandle=self.Function(lambda h:(self.closed.append(h) or True))
    def _exit(self,h,p):p._obj.value=259 if self.active else 1;return True

def test_windows_pid_helper_null_inactive_64bit_and_close():
    assert not _pid_alive_windows(1,Kernel(0))
    k=Kernel(2**40,False);assert not _pid_alive_windows(1,k) and k.closed==[2**40]
    k=Kernel(2**40,True);assert _pid_alive_windows(1,k) and k.closed==[2**40]

def test_posix_pid_helper_uses_signal_zero():
    calls=[];assert _pid_alive_posix(7,lambda p,s:calls.append((p,s))) and calls==[(7,0)]
    assert not _pid_alive_posix(7,lambda *_:(_ for _ in ()).throw(ProcessLookupError()))

def coordinator(tmp_path,trace,**overrides):
    path=tmp_path/"db.sqlite3"
    def dbf():trace.append("database");return Database(path)
    values=dict(database_factory=dbf,marker_path=tmp_path/"ready",lock=InstanceLock(tmp_path/"lock",clock=lambda:NOW),owner_qq=OWNER,bot_qq=BOT,model_capability=capability(),identity_provider=lambda:(trace.append("identity") or IdentityEvidence(OWNER,BOT,"id")),ws_provider=lambda:(trace.append("ws") or WebSocketEvidence("ws",OWNER,BOT,True)),memory_worker_provider=lambda:(trace.append("memory") or WorkerEvidence("mem",True,True)),initiative_worker_provider=lambda:(trace.append("initiative") or WorkerEvidence("init",True,True)),clock=lambda:NOW)
    values.update(overrides);return ReadinessCoordinator(**values)

def test_coordinator_success_order_marker_and_stop(tmp_path):
    trace=[];c=coordinator(tmp_path,trace);m=c.start();assert trace==["database","identity","ws","memory","initiative"] and m.last_recovered_sequence==-1
    assert (m.version,m.provider,m.model,m.context_window)==(READY_MARKER_VERSION,"p","m",262144)
    assert m.build_id==BUILD_ID and json.loads((tmp_path/"ready").read_text())["build_id"]==BUILD_ID
    assert ReadyMarker.from_dict(m.to_dict())==m
    assert (tmp_path/"ready").exists() and (tmp_path/"lock").exists();c.stop();assert not (tmp_path/"ready").exists() and not (tmp_path/"lock").exists()

def test_legacy_v1_marker_remains_readable_without_claiming_model_facts(tmp_path):
    record=InstanceLock(tmp_path/"lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    legacy=ReadyMarker.from_dict(legacy_marker(record,1))
    assert legacy.version==1
    assert legacy.provider is None and legacy.model is None and legacy.context_window is None
    assert set(legacy.to_dict())==set(ReadyMarker.__dataclass_fields__)-{"provider","model","context_window","build_id"}

def test_legacy_v1_marker_rejects_v2_model_fields(tmp_path):
    record=InstanceLock(tmp_path/"lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw=legacy_marker(record,1);raw["provider"]="p"
    with pytest.raises(ReadinessError,match="schema is not exact"):
        ReadyMarker.from_dict(raw)

@pytest.mark.parametrize("missing",["provider","model","context_window"])
def test_ready_v2_requires_every_model_field(tmp_path,missing):
    record=InstanceLock(tmp_path/"lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw=legacy_marker(record,2);raw.pop(missing)
    with pytest.raises(ReadinessError,match="schema is not exact"):
        ReadyMarker.from_dict(raw)

@pytest.mark.parametrize(("field","invalid"),[("provider",""),("provider",None),("model",""),("model",None),("context_window",0),("context_window",-1),("context_window",True),("context_window","262144")])
def test_ready_v2_rejects_invalid_model_facts(tmp_path,field,invalid):
    record=InstanceLock(tmp_path/"lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw=legacy_marker(record,2);raw[field]=invalid
    with pytest.raises(ReadinessError):
        ReadyMarker.from_dict(raw)

def test_coordinator_accepts_configured_128k_context_gate(tmp_path):
    trace=[]
    c=coordinator(tmp_path,trace,model_capability=capability(131072),required_context_tokens=131072)
    marker_result=c.start()
    assert marker_result.last_recovered_sequence == -1
    c.stop()

def test_coordinator_rejects_model_below_configured_128k_gate(tmp_path):
    trace=[]
    c=coordinator(tmp_path,trace,model_capability=capability(65536),required_context_tokens=131072)
    with pytest.raises(Exception):
        c.start()
    assert trace == [] and not (tmp_path/"lock").exists()

@pytest.mark.parametrize("field",["identity_provider","ws_provider","memory_worker_provider","initiative_worker_provider"])
def test_coordinator_each_evidence_gate_fails_without_artifacts(tmp_path,field):
    trace=[];c=coordinator(tmp_path,trace,**{field:lambda:True})
    with pytest.raises(ReadinessError):c.start()
    assert not (tmp_path/"ready").exists() and not (tmp_path/"lock").exists()

def test_coordinator_model_unknown_fails_before_lock(tmp_path):
    trace=[];c=coordinator(tmp_path,trace,model_capability=ModelCapability("m",262144,"p",None))
    with pytest.raises(Exception):c.start()
    assert trace==[] and not (tmp_path/"lock").exists()

def test_concurrent_foreign_lock_is_never_replaced_or_removed(tmp_path):
    path=tmp_path/"lock";first=InstanceLock(path,instance_id="first",clock=lambda:NOW);first.acquire(pid=10,parent_pid=11);before=path.read_bytes()
    second=InstanceLock(path,instance_id="second",clock=lambda:NOW)
    with pytest.raises(ReadinessError):second.acquire(pid=12,parent_pid=13)
    assert path.read_bytes()==before and second.release() is False and path.exists()

def test_reclaim_stale_runtime_artifacts_archives_matching_dead_pair(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,instance_id="old",lock_identity="old-lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    write_ready_marker(marker_path,marker(record),lock_path=lock_path)

    archived=reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)

    assert not lock_path.exists() and not marker_path.exists()
    assert len(archived)==2 and all(path.exists() and ".stale-" in path.name for path in archived)

def test_reclaim_stale_runtime_artifacts_archives_dead_legacy_v2_marker_after_database_migration(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,instance_id="old",lock_identity="old-lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw={"schema":"qichi-ready","version":2,"instance_id":record.instance_id,"lock_identity":record.lock_identity,
         "pid":record.pid,"parent_pid":record.parent_pid,"owner_qq":OWNER,"bot_qq":BOT,
         "database_schema_version":2,"last_recovered_sequence":-1,"ws_connection_id":"ws",
         "memory_worker_id":"mem","initiative_worker_id":"init","started_at_utc":NOW.isoformat(),
         "provider":"p","model":"m","context_window":262144}
    marker_path.write_text(json.dumps(raw),encoding="utf-8")

    archived=reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)

    assert len(archived)==2 and not marker_path.exists() and not lock_path.exists()

def test_reclaim_stale_runtime_artifacts_archives_dead_previous_schema_three_marker(tmp_path):
    """A dead marker from the immediately previous DB schema must not block upgrade."""
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,instance_id="old",lock_identity="old-lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw={"schema":"qichi-ready","version":2,"instance_id":record.instance_id,"lock_identity":record.lock_identity,
         "pid":record.pid,"parent_pid":record.parent_pid,"owner_qq":OWNER,"bot_qq":BOT,
         "database_schema_version":SCHEMA_VERSION - 1,"last_recovered_sequence":-1,"ws_connection_id":"ws",
         "memory_worker_id":"mem","initiative_worker_id":"init","started_at_utc":NOW.isoformat(),
         "provider":"p","model":"m","context_window":262144}
    marker_path.write_text(json.dumps(raw),encoding="utf-8")

    archived=reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)

    assert len(archived)==2 and not marker_path.exists() and not lock_path.exists()

def test_reclaim_stale_runtime_artifacts_rejects_future_schema_marker(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,instance_id="old",lock_identity="old-lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw={"schema":"qichi-ready","version":2,"instance_id":record.instance_id,"lock_identity":record.lock_identity,
         "pid":record.pid,"parent_pid":record.parent_pid,"owner_qq":OWNER,"bot_qq":BOT,
         "database_schema_version":SCHEMA_VERSION + 1,"last_recovered_sequence":-1,"ws_connection_id":"ws",
         "memory_worker_id":"mem","initiative_worker_id":"init","started_at_utc":NOW.isoformat(),
         "provider":"p","model":"m","context_window":262144}
    marker_path.write_text(json.dumps(raw),encoding="utf-8")
    before=(lock_path.read_bytes(),marker_path.read_bytes())

    with pytest.raises(ReadinessError,match="READY marker invalid"):
        reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)
    assert (lock_path.read_bytes(),marker_path.read_bytes())==before

def test_check_ready_rejects_legacy_v2_database_schema(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,instance_id="old",lock_identity="old-lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw={"schema":"qichi-ready","version":2,"instance_id":record.instance_id,"lock_identity":record.lock_identity,
         "pid":record.pid,"parent_pid":record.parent_pid,"owner_qq":OWNER,"bot_qq":BOT,
         "database_schema_version":2,"last_recovered_sequence":-1,"ws_connection_id":"ws",
         "memory_worker_id":"mem","initiative_worker_id":"init","started_at_utc":NOW.isoformat(),
         "provider":"p","model":"m","context_window":262144}
    marker_path.write_text(json.dumps(raw),encoding="utf-8")

    with pytest.raises(ReadinessError,match="marker invalid"):
        check_ready(marker_path,lock_path,expected_owner=OWNER,expected_bot=BOT,pid_probe=lambda _pid:True)

def test_reclaim_stale_runtime_artifacts_rejects_malformed_marker(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,instance_id="old",lock_identity="old-lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    marker_path.write_text('{"schema":"qichi-ready","version":2}',encoding="utf-8")
    before=(lock_path.read_bytes(),marker_path.read_bytes())

    with pytest.raises(ReadinessError,match="READY marker invalid"):
        reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)
    assert (lock_path.read_bytes(),marker_path.read_bytes())==before

@pytest.mark.parametrize("orphan",["lock","marker"])
def test_reclaim_stale_runtime_artifacts_archives_dead_orphan(tmp_path,orphan):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    write_ready_marker(marker_path,marker(record),lock_path=lock_path)
    (marker_path if orphan=="lock" else lock_path).unlink()

    archived=reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)

    target=lock_path if orphan=="lock" else marker_path
    assert not target.exists() and len(archived)==1 and archived[0].exists()

def test_reclaim_stale_runtime_artifacts_preserves_live_pair(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    write_ready_marker(marker_path,marker(record),lock_path=lock_path)
    before=(lock_path.read_bytes(),marker_path.read_bytes())

    assert reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:True)==()
    assert (lock_path.read_bytes(),marker_path.read_bytes())==before


def test_reclaim_runtime_probe_rejects_reused_pid_even_when_it_is_alive(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    write_ready_marker(marker_path,marker(record),lock_path=lock_path)
    probes=[]

    def probe(pid, started):
        probes.append((pid, started))
        return False

    archived=reclaim_stale_runtime_artifacts(
        marker_path,
        lock_path,
        pid_probe=lambda _pid:True,
        runtime_probe=probe,
    )

    assert len(archived)==2
    assert [pid for pid, _started in probes] == [10, 11]
    assert all(started == NOW.isoformat() for _pid, started in probes)

def test_reclaim_stale_runtime_artifacts_rejects_mismatched_pair(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    record=InstanceLock(lock_path,clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    write_ready_marker(marker_path,marker(record),lock_path=lock_path)
    raw=json.loads(marker_path.read_text());raw["instance_id"]="other";marker_path.write_text(json.dumps(raw))
    before=(lock_path.read_bytes(),marker_path.read_bytes())

    with pytest.raises(ReadinessError,match="mismatch"):
        reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)
    assert (lock_path.read_bytes(),marker_path.read_bytes())==before

def test_reclaim_stale_runtime_artifacts_rejects_invalid_lock(tmp_path):
    lock_path=tmp_path/"qichi.lock";marker_path=tmp_path/"qichi-ready.json"
    lock_path.write_text("not-json",encoding="utf-8")
    before=lock_path.read_bytes()

    with pytest.raises(ReadinessError,match="lock invalid"):
        reclaim_stale_runtime_artifacts(marker_path,lock_path,pid_probe=lambda _pid:False)
    assert lock_path.read_bytes()==before and not marker_path.exists()

def test_coordinator_atomic_marker_failure_closes_database_and_cleans_own_lock(tmp_path,monkeypatch):
    trace=[]
    class TrackedDatabase(Database):
        def close(self):trace.append("close");super().close()
    def dbf():return TrackedDatabase(tmp_path/"db.sqlite3")
    c=coordinator(tmp_path,trace,database_factory=dbf)
    monkeypatch.setattr(os,"replace",lambda *_:(_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError):c.start()
    assert "close" in trace and not (tmp_path/"ready").exists() and not (tmp_path/"lock").exists()

def test_start_stack_cli_refuses_without_writing_marker(tmp_path):
    marker=tmp_path/"ready.json"
    result=subprocess.run([sys.executable,"scripts/start_stack.py",str(marker)],cwd=Path(__file__).parents[1],capture_output=True,text=True)
    assert result.returncode!=0 and not marker.exists()

def test_check_ready_cli_uses_marker_then_lock_paths(tmp_path):
    marker_path=tmp_path/"ready";lock_path=tmp_path/"lock";record=InstanceLock(lock_path,clock=lambda:NOW).acquire(pid=os.getpid(),parent_pid=os.getpid());write_ready_marker(marker_path,marker(record),lock_path=lock_path)
    root=Path(__file__).parents[1]
    ok=subprocess.run([sys.executable,"scripts/check_ready.py",str(marker_path),str(lock_path),"--owner",OWNER,"--bot",BOT],cwd=root,capture_output=True,text=True)
    swapped=subprocess.run([sys.executable,"scripts/check_ready.py",str(lock_path),str(marker_path),"--owner",OWNER,"--bot",BOT],cwd=root,capture_output=True,text=True)
    assert ok.returncode==0 and swapped.returncode!=0

def test_build_evidence_is_deterministic_and_ignores_non_runtime_files(tmp_path):
    runtime_tree(tmp_path)
    first=compute_build_id(tmp_path)
    assert first==compute_build_id(tmp_path) and len(first)==64
    (tmp_path/"doc").mkdir();(tmp_path/"doc"/"note.md").write_text("changed\n",encoding="utf-8")
    (tmp_path/"dashboard").mkdir();(tmp_path/"dashboard"/"app.js").write_text("console.log(1)\n",encoding="utf-8")
    assert compute_build_id(tmp_path)==first
    (tmp_path/"src"/"qichi"/"app.py").write_text("x=2\n",encoding="utf-8")
    changed_source=compute_build_id(tmp_path)
    assert changed_source!=first
    (tmp_path/"migrations"/"0002_x.sql").write_text("CREATE TABLE u(x);\n",encoding="utf-8")
    assert compute_build_id(tmp_path) not in {first,changed_source}

def test_build_evidence_ignores_the_read_only_dashboard_package(tmp_path):
    runtime_tree(tmp_path)
    first=compute_build_id(tmp_path)
    dash=tmp_path/"src"/"qichi"/"dashboard";dash.mkdir(parents=True)
    (dash/"service.py").write_text("x=1\n",encoding="utf-8")
    assert compute_build_id(tmp_path)==first
    (dash/"service.py").write_text("x=2\n",encoding="utf-8")
    assert compute_build_id(tmp_path)==first,"只读面板的改动不该要求重启机器人"
    (tmp_path/"src"/"qichi"/"app.py").write_text("x=3\n",encoding="utf-8")
    assert compute_build_id(tmp_path)!=first,"回复链路改动仍然必须重启"

def test_build_evidence_fails_closed_without_runtime_sources(tmp_path):
    with pytest.raises(ReadinessError):compute_build_id(tmp_path)

def test_check_ready_accepts_current_build_and_rejects_other_code(tmp_path):
    lp=tmp_path/"lock";mp=tmp_path/"ready";record=InstanceLock(lp,clock=lambda:NOW).acquire(pid=os.getpid(),parent_pid=os.getpid())
    write_ready_marker(mp,marker(record),lock_path=lp)
    assert check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,expected_build_id=BUILD_ID,pid_probe=lambda p:True).build_id==BUILD_ID
    with pytest.raises(ReadinessError,match="build evidence mismatch"):
        check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,expected_build_id="0"*64,pid_probe=lambda p:True)
    with pytest.raises(ReadinessError):check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,expected_build_id="not-a-digest",pid_probe=lambda p:True)

def test_check_ready_rejects_legacy_marker_without_build_evidence(tmp_path):
    lp=tmp_path/"lock";mp=tmp_path/"ready";record=InstanceLock(lp,clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    mp.write_text(json.dumps(legacy_marker(record,2)),encoding="utf-8")
    with pytest.raises(ReadinessError,match="marker build evidence missing"):
        check_ready(mp,lp,expected_owner=OWNER,expected_bot=BOT,pid_probe=lambda p:True)

def test_current_marker_version_requires_exact_build_evidence(tmp_path):
    record=InstanceLock(tmp_path/"lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    raw=marker(record).to_dict();raw.pop("build_id")
    with pytest.raises(ReadinessError,match="schema is not exact"):ReadyMarker.from_dict(raw)
    for invalid in ("","not-a-digest","0"*63,"0"*65,"G"*64,None,123):
        raw=marker(record).to_dict();raw["build_id"]=invalid
        with pytest.raises(ReadinessError):ReadyMarker.from_dict(raw)

def test_legacy_marker_versions_reject_build_evidence_field(tmp_path):
    record=InstanceLock(tmp_path/"lock",clock=lambda:NOW).acquire(pid=10,parent_pid=11)
    for version in (1,2):
        raw=legacy_marker(record,version);raw["build_id"]=BUILD_ID
        with pytest.raises(ReadinessError,match="schema is not exact"):ReadyMarker.from_dict(raw)
