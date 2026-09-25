"""Fail-closed local recovery and readiness evidence."""
from __future__ import annotations
import ctypes, hashlib, json, os, tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4
from qichi.dialogue.model_capability import ModelCapability
from qichi.storage.database import Database
from qichi.storage.migrations import SCHEMA_VERSION
from qichi.storage.outbox_repository import OutboxRepository

class ReadinessError(RuntimeError): pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# READY is only valid for the exact source tree that produced it.  The marker
# therefore carries a build digest, and every reader compares that digest with
# the code it is actually running instead of trusting a bare schema number.
READY_MARKER_VERSION = 3
_BUILD_EVIDENCE_MIN_VERSION = 3
BUILD_ID_LENGTH = 64

# Files that define runtime behaviour.  Documentation, the dashboard frontend
# assets and the launcher scripts may change without invalidating a running
# marker.  The whole qichi package is covered -- including the read-only
# dashboard sidecar, which starts and stops with the production stack -- so any
# package edit requires a restart to clear the warning instead of passing
# silently; tests and quality fixtures are not part of the running code.
_RUNTIME_BUILD_SOURCES = ("src/qichi/**/*.py", "migrations/*.sql")
# The read-only dashboard sidecar runs with the stack but is not on the reply path:
# keeping it out of the fingerprint means panel iteration no longer needs a bot
# restart, while anything the bot itself executes still invalidates a live marker.
_RUNTIME_BUILD_EXCLUDES = ("src/qichi/dashboard/",)


def compute_build_id(root: str | Path) -> str:
    """Return the deterministic build digest for one Qichi source tree."""
    base = Path(root)
    entries: list[str] = []
    for pattern in _RUNTIME_BUILD_SOURCES:
        for path in sorted(base.glob(pattern), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            relative = path.relative_to(base).as_posix()
            if any(relative.startswith(prefix) for prefix in _RUNTIME_BUILD_EXCLUDES):
                continue
            entries.append(f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    if not entries:
        raise ReadinessError("build evidence root holds no runtime source")
    payload = f"qichi-build-v1\nschema={SCHEMA_VERSION}\n" + "".join(entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != BUILD_ID_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReadinessError(f"{field} must be a lowercase sha256 digest")
    return value


def current_build_id() -> str:
    """Build evidence for the code that is executing right now."""
    return compute_build_id(PROJECT_ROOT)


def marker_build_reason(marker: "ReadyMarker", *, expected_build_id: str) -> str | None:
    """Say why a marker cannot vouch for the code that is running now.

    One rule with two presentations: check_ready raises this reason and the
    dashboard reports it as the health reason.
    """
    _build_id(expected_build_id, "expected_build_id")
    if marker.version < _BUILD_EVIDENCE_MIN_VERSION or marker.build_id is None:
        return "marker build evidence missing"
    if marker.build_id != expected_build_id:
        return "build evidence mismatch"
    return None

def _utc(value: object, field: str) -> str:
    if not isinstance(value, str): raise ReadinessError(f"{field} must be UTC text")
    try: parsed=datetime.fromisoformat(value)
    except ValueError as e: raise ReadinessError(f"{field} is invalid") from e
    if parsed.tzinfo is None or parsed.utcoffset()!=timezone.utc.utcoffset(parsed): raise ReadinessError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc).isoformat()

def _decimal(value: object, field: str) -> str:
    if not isinstance(value,str) or not value or not value.isdecimal(): raise ReadinessError(f"{field} must be decimal")
    return value

def _pid(value: object, field: str) -> int:
    if type(value) is not int or value<=0: raise ReadinessError(f"{field} is invalid")
    return value

def _exact(value: Mapping[str,object], fields: set[str], label: str)->None:
    if not isinstance(value,Mapping) or set(value)!=fields: raise ReadinessError(f"{label} schema is not exact")

@dataclass(frozen=True,slots=True)
class LockRecord:
    schema:str; version:int; instance_id:str; lock_identity:str; pid:int; parent_pid:int; acquired_at_utc:str
    @classmethod
    def from_dict(cls,value:Mapping[str,object])->"LockRecord":
        _exact(value,set(cls.__dataclass_fields__),"lock")
        if value["schema"]!="qichi-lock" or type(value["version"]) is not int or value["version"]!=1: raise ReadinessError("lock version unsupported")
        for f in ("instance_id","lock_identity"):
            if not isinstance(value[f],str) or not value[f]: raise ReadinessError(f"{f} invalid")
        _pid(value["pid"],"pid"); _pid(value["parent_pid"],"parent_pid"); _utc(value["acquired_at_utc"],"acquired_at_utc")
        return cls(**value)  # type: ignore[arg-type]
    def to_dict(self)->dict[str,object]: return asdict(self)

@dataclass(frozen=True,slots=True)
class ReadyMarker:
    schema:str; version:int; instance_id:str; lock_identity:str; pid:int; parent_pid:int; owner_qq:str; bot_qq:str
    database_schema_version:int; last_recovered_sequence:int; ws_connection_id:str; memory_worker_id:str; initiative_worker_id:str; started_at_utc:str
    provider:str|None=None; model:str|None=None; context_window:int|None=None
    build_id:str|None=None
    @classmethod
    def from_dict(cls,value:Mapping[str,object], *, allow_legacy_database_schema:bool=False)->"ReadyMarker":
        if not isinstance(value,Mapping):raise ReadinessError("READY marker schema is not exact")
        version=value.get("version")
        if type(version) is not int or version not in {1,2,READY_MARKER_VERSION}:raise ReadinessError("READY version unsupported")
        model_fields={"provider","model","context_window"}
        expected_fields=set(cls.__dataclass_fields__)
        if version==1:expected_fields-=model_fields
        if version<READY_MARKER_VERSION:expected_fields-={"build_id"}
        _exact(value,expected_fields,"READY marker")
        if value["schema"]!="qichi-ready":raise ReadinessError("READY schema unsupported")
        for f in ("instance_id","lock_identity","ws_connection_id","memory_worker_id","initiative_worker_id"):
            if not isinstance(value[f],str) or not value[f]: raise ReadinessError(f"{f} invalid")
        _pid(value["pid"],"pid"); _pid(value["parent_pid"],"parent_pid")
        owner=_decimal(value["owner_qq"],"owner_qq"); bot=_decimal(value["bot_qq"],"bot_qq")
        if owner==bot: raise ReadinessError("owner and bot must differ")
        database_schema_version=value["database_schema_version"]
        # Strict READY checks only accept the running schema.  Stale-artifact
        # reclamation is different: a structurally valid marker from any
        # previously supported database schema may be archived after its
        # processes are proven dead, while a future schema must remain
        # fail-closed so we never delete evidence we cannot interpret.
        legacy_schema_allowed=(
            allow_legacy_database_schema
            and type(database_schema_version) is int
            and 1 <= database_schema_version <= SCHEMA_VERSION
        )
        if type(database_schema_version) is not int or (database_schema_version!=SCHEMA_VERSION and not legacy_schema_allowed): raise ReadinessError("schema version unsupported")
        if type(value["last_recovered_sequence"]) is not int or value["last_recovered_sequence"] < -1: raise ReadinessError("recovery sequence invalid")
        _utc(value["started_at_utc"],"started_at_utc")
        parsed=dict(value)
        if version==1:
            parsed.update({field:None for field in model_fields})
        else:
            for field in ("provider","model"):
                if not isinstance(value[field],str) or not value[field]:raise ReadinessError(f"{field} invalid")
            if type(value["context_window"]) is not int or value["context_window"]<=0:raise ReadinessError("context_window invalid")
        if version<READY_MARKER_VERSION:
            parsed["build_id"]=None
        else:
            _build_id(value["build_id"],"build_id")
        return cls(**parsed)  # type: ignore[arg-type]
    def to_dict(self)->dict[str,object]:
        value=asdict(self)
        if self.version not in {1,2,READY_MARKER_VERSION}:raise ReadinessError("READY version unsupported")
        if self.version==1:
            if any(value[field] is not None for field in ("provider","model","context_window")):raise ReadinessError("READY v1 model facts invalid")
            for field in ("provider","model","context_window"):value.pop(field)
        if self.version<READY_MARKER_VERSION:
            if value["build_id"] is not None:raise ReadinessError("READY build evidence requires the current marker version")
            value.pop("build_id")
        return value

@dataclass(frozen=True,slots=True)
class IdentityEvidence:
    owner_qq:str; bot_qq:str; evidence_id:str
    def __post_init__(self)->None:
        if _decimal(self.owner_qq,"owner_qq")==_decimal(self.bot_qq,"bot_qq") or not isinstance(self.evidence_id,str) or not self.evidence_id: raise ReadinessError("identity evidence invalid")

@dataclass(frozen=True,slots=True)
class WebSocketEvidence:
    connection_id:str; authenticated_owner_qq:str; authenticated_bot_qq:str; unique_consumer:bool
    def __post_init__(self)->None:
        if not isinstance(self.connection_id,str) or not self.connection_id or type(self.unique_consumer) is not bool or not self.unique_consumer: raise ReadinessError("websocket evidence invalid")
        if _decimal(self.authenticated_owner_qq,"owner_qq")==_decimal(self.authenticated_bot_qq,"bot_qq"): raise ReadinessError("websocket identities invalid")

@dataclass(frozen=True,slots=True)
class WorkerEvidence:
    worker_id:str; recovered:bool; alive:bool
    def __post_init__(self)->None:
        if not isinstance(self.worker_id,str) or not self.worker_id or self.recovered is not True or self.alive is not True: raise ReadinessError("worker evidence invalid")

@dataclass(frozen=True,slots=True)
class RecoveryReport:
    last_recovered_sequence:int; pending_text:tuple[str,...]; pending_reaction:tuple[str,...]; frozen_unknown:tuple[str,...]; frozen_dispatched:tuple[str,...]; terminal:tuple[str,...]; network_calls:int=0

def recover_database(database:Database,*,conversation_id:str)->RecoveryReport:
    if not isinstance(database,Database): raise TypeError("database must be Database")
    owner=_decimal(conversation_id,"conversation_id")
    _verify_database(database)
    cursor=database.connection.execute("SELECT last_processed_sequence FROM conversation_cursors WHERE conversation_id=?",(owner,)).fetchone()
    last=-1 if cursor is None or cursor[0] is None else cursor[0]
    if type(last) is not int or last < -1: raise ReadinessError("processed cursor invalid")
    pending=database.connection.execute("SELECT 1 FROM conversation_events WHERE conversation_id=? AND direction='inbound' AND status IN ('received','processing') AND sequence>? LIMIT 1",(owner,last)).fetchone()
    if pending is not None: raise ReadinessError("unprocessed inbound requires explicit replay")
    buckets={k:[] for k in ("pending_text","pending_reaction","frozen_unknown","frozen_dispatched","terminal")}
    outbox = OutboxRepository(database)
    keys = database.connection.execute("SELECT operation_key FROM outbox ORDER BY operation_key").fetchall()
    for key_row in keys:
        key = key_row["operation_key"]
        try:
            record = outbox.get(key)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ReadinessError("outbox record is invalid") from error
        action=record.payload.get("action_kind"); status=record.status
        # 2026-09-15：record（QQ 语音）也必须在这里被承认 —— 漏了它，一条没发完的语音会让
        # 整个启动 fail closed（真机上就是这么起不来的，而且被 stop() 的另一个 bug 掩盖）。
        # 语音的契约是"永不重发"（超时/断连一律 unknown；重复一条语音比少一条更糟），
        # 所以它的 pending 既不进 pending_text 那条重发通道，也不因为 attempt_count 而 raise：
        # 一条语音绝不该把启动卡死。它收进 frozen，只作诊断。
        if action not in {"text","face","reaction","poke","record"}: raise ReadinessError("outbox action invalid")
        if status=="pending" and action == "record":
            buckets["frozen_dispatched"].append(key)
            continue
        if status=="pending":
            # A reaction set=true may be safely returned to the idempotent
            # recovery path after a prior dispatch attempt; other actions
            # with attempts are ambiguous and remain fail-closed.
            if record.attempt_count != 0 and not (action == "reaction" and record.payload.get("set") is True):
                raise ReadinessError("pending outbox ambiguous")
            buckets["pending_reaction" if action=="reaction" else "pending_text"].append(key)
        elif status=="unknown": buckets["frozen_unknown"].append(key)
        elif status=="dispatched": buckets["frozen_dispatched"].append(key)
        elif status in {"sent","failed"}: buckets["terminal"].append(key)
        else: raise ReadinessError("outbox status invalid")
    return RecoveryReport(last,*(tuple(buckets[k]) for k in ("pending_text","pending_reaction","frozen_unknown","frozen_dispatched","terminal")))

def _verify_database(database:Database)->None:
    rows=database.connection.execute("PRAGMA integrity_check").fetchall()
    if len(rows)!=1 or tuple(rows[0])!=("ok",): raise ReadinessError("integrity check failed")
    schema=database.connection.execute("SELECT value_json FROM runtime_meta WHERE key='schema_version'").fetchone()
    if schema is None or schema[0]!=str(SCHEMA_VERSION): raise ReadinessError("runtime schema is not current")

def _atomic_json(path:Path,value:Mapping[str,object])->None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as h: json.dump(value,h,sort_keys=True,separators=(",",":")); h.flush(); os.fsync(h.fileno())
        os.replace(temp,path)
    except BaseException:
        try: os.unlink(temp)
        except FileNotFoundError: pass
        raise

def _read(path:Path,parser:Callable[[Mapping[str,object]],Any],label:str)->Any:
    try:
        raw=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw,Mapping): raise ReadinessError(f"{label} invalid")
        return parser(raw)
    except (OSError,json.JSONDecodeError,ReadinessError) as e: raise ReadinessError(f"{label} invalid") from e

class InstanceLock:
    def __init__(self,path:str|Path,*,instance_id:str|None=None,lock_identity:str|None=None,clock:Callable[[],datetime]|None=None):
        self.path=Path(path); self.instance_id=instance_id or uuid4().hex; self.lock_identity=lock_identity or uuid4().hex; self.clock=clock or (lambda:datetime.now(timezone.utc)); self.record:LockRecord|None=None
    def acquire(self,*,pid:int|None=None,parent_pid:int|None=None)->LockRecord:
        record=LockRecord.from_dict({"schema":"qichi-lock","version":1,"instance_id":self.instance_id,"lock_identity":self.lock_identity,"pid":os.getpid() if pid is None else pid,"parent_pid":os.getppid() if parent_pid is None else parent_pid,"acquired_at_utc":self.clock().astimezone(timezone.utc).isoformat()})
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try: fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError as e: raise ReadinessError("instance lock exists") from e
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as h: json.dump(record.to_dict(),h,sort_keys=True,separators=(",",":")); h.flush(); os.fsync(h.fileno())
        except BaseException:
            try:
                current = _read(self.path, LockRecord.from_dict, "lock")
                if current == record:
                    self.path.unlink()
            except (FileNotFoundError, ReadinessError):
                pass
            raise
        self.record=record; return record
    def release(self)->bool:
        if self.record is None:return False
        try:current=_read(self.path,LockRecord.from_dict,"lock")
        except ReadinessError:self.record=None;return False
        if current!=self.record:self.record=None;return False
        self.path.unlink();self.record=None;return True

def write_ready_marker(path:str|Path,marker:ReadyMarker,*,lock_path:str|Path)->None:
    if not isinstance(marker,ReadyMarker):raise TypeError("marker must be ReadyMarker")
    lock=_read(Path(lock_path),LockRecord.from_dict,"lock")
    if any(getattr(marker,f)!=getattr(lock,f) for f in ("instance_id","lock_identity","pid","parent_pid")):raise ReadinessError("marker does not match lock")
    target=Path(path)
    if target.exists():
        current=_read(target,ReadyMarker.from_dict,"marker")
        if current!=marker:raise ReadinessError("foreign READY marker exists")
    # Re-parse the serialized representation so the exact schema is enforced
    # at the persistence boundary as well as at construction time.
    validated = ReadyMarker.from_dict(marker.to_dict())
    _atomic_json(target,validated.to_dict())

def _pid_alive_windows(pid:int,kernel:Any,ctypes_api:Any=ctypes)->bool:
    kernel.OpenProcess.argtypes=[ctypes_api.c_ulong,ctypes_api.c_bool,ctypes_api.c_ulong];kernel.OpenProcess.restype=ctypes_api.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes_api.c_void_p,ctypes_api.POINTER(ctypes_api.c_ulong)];kernel.GetExitCodeProcess.restype=ctypes_api.c_bool
    kernel.CloseHandle.argtypes=[ctypes_api.c_void_p];kernel.CloseHandle.restype=ctypes_api.c_bool
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        code=ctypes_api.c_ulong();return bool(kernel.GetExitCodeProcess(handle,ctypes_api.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)

def _pid_alive_posix(pid:int,kill:Callable[[int,int],Any]=os.kill)->bool:
    try:kill(pid,0)
    except OSError:return False
    return True

def pid_alive(pid:int)->bool:
    if type(pid) is not int or pid<=0:return False
    return _pid_alive_windows(pid,ctypes.WinDLL("kernel32",use_last_error=True)) if os.name=="nt" else _pid_alive_posix(pid)


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


def _filetime_datetime(value: _FileTime) -> datetime:
    ticks = (int(value.high) << 32) | int(value.low)
    return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks / 10)


def _windows_process_evidence(pid: int) -> tuple[str, datetime] | None:
    """Return the executable and creation time for one Windows PID.

    READY recovery must distinguish a dead Qichi process from a later process
    that happens to reuse its PID.  Python's standard library has no portable
    process identity API, so use the read-only Windows query APIs here and
    fail closed when the process cannot be inspected.
    """
    if os.name != "nt" or type(pid) is not int or pid <= 0:
        return None
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetProcessTimes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        kernel.GetProcessTimes.restype = ctypes.c_bool
        kernel.QueryFullProcessImageNameW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_wchar),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel.QueryFullProcessImageNameW.restype = ctypes.c_bool
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_bool
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            created, exited, kernel_time, user_time = (_FileTime() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            capacity = ctypes.c_ulong(32768)
            buffer = ctypes.create_unicode_buffer(capacity.value)
            if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(capacity)):
                return None
            return buffer.value, _filetime_datetime(created)
        finally:
            kernel.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def runtime_process_probe(pid: int, *, acquired_at_utc: str, expected_executable: str) -> bool:
    """Check that a live PID is the Qichi executable from this runtime.

    ``acquired_at_utc`` is the lower-bound evidence written before READY.  A
    process created after that timestamp is a PID reuse, even if it has the
    same executable path.  This function is intentionally narrow and read-only
    so it can only improve lifecycle evidence; it never kills or restarts a
    process.
    """
    if not pid_alive(pid):
        return False
    if os.name != "nt":
        try:
            return os.path.samefile(f"/proc/{pid}/exe", expected_executable)
        except (FileNotFoundError, OSError):
            return False
    evidence = _windows_process_evidence(pid)
    if evidence is None:
        return False
    actual_path, created_at = evidence
    try:
        acquired = datetime.fromisoformat(acquired_at_utc).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False
    return (
        os.path.normcase(os.path.normpath(actual_path))
        == os.path.normcase(os.path.normpath(expected_executable))
        and created_at <= acquired
    )

def reclaim_stale_runtime_artifacts(marker_path:str|Path,lock_path:str|Path,*,pid_probe:Callable[[int],bool]=pid_alive,runtime_probe:Callable[[int,str],bool]|None=None)->tuple[Path,...]:
    """Archive only structurally valid runtime evidence owned by dead processes."""
    if not callable(pid_probe):raise TypeError("pid_probe must be callable")
    marker_target=Path(marker_path);lock_target=Path(lock_path)
    if not marker_target.exists() and not lock_target.exists():return ()
    lock_target.parent.mkdir(parents=True,exist_ok=True)
    recovery_guard=lock_target.with_name(lock_target.name+".recovery")
    try:guard_fd=os.open(recovery_guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as e:raise ReadinessError("runtime artifact recovery already in progress") from e
    try:
        with os.fdopen(guard_fd,"w",encoding="ascii") as handle:
            handle.write(str(os.getpid()));handle.flush();os.fsync(handle.fileno())
        lock=_read(lock_target,LockRecord.from_dict,"lock") if lock_target.exists() else None
        marker_parser=lambda raw: ReadyMarker.from_dict(raw, allow_legacy_database_schema=True)
        marker=_read(marker_target,marker_parser,"READY marker") if marker_target.exists() else None
        if lock is not None and marker is not None:
            fields=("instance_id","lock_identity","pid","parent_pid")
            if any(getattr(lock,field)!=getattr(marker,field) for field in fields):
                raise ReadinessError("runtime marker/lock mismatch")
        evidence=lock if lock is not None else marker
        if evidence is None:return ()
        probe = runtime_probe or (lambda pid, _started: pid_probe(pid))
        started_at = lock.acquired_at_utc if lock is not None else marker.started_at_utc
        if probe(evidence.pid, started_at) or probe(evidence.parent_pid, started_at):
            if lock is None or marker is None:
                raise ReadinessError("live runtime evidence is incomplete")
            return ()
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archived=[]
        for target in (marker_target,lock_target):
            if not target.exists():continue
            archive=target.with_name(f"{target.name}.stale-{stamp}-{uuid4().hex[:8]}")
            target.replace(archive);archived.append(archive)
        return tuple(archived)
    finally:
        try:recovery_guard.unlink()
        except FileNotFoundError:pass

def check_ready(marker_path:str|Path,lock_path:str|Path,*,expected_owner:str,expected_bot:str,expected_build_id:str|None=None,pid_probe:Callable[[int],bool]=pid_alive)->ReadyMarker:
    owner=_decimal(expected_owner,"expected_owner");bot=_decimal(expected_bot,"expected_bot")
    if owner==bot:raise ReadinessError("expected identities must differ")
    marker=_read(Path(marker_path),ReadyMarker.from_dict,"marker");lock=_read(Path(lock_path),LockRecord.from_dict,"lock")
    build_reason=marker_build_reason(marker,expected_build_id=current_build_id() if expected_build_id is None else expected_build_id)
    if build_reason is not None:raise ReadinessError(build_reason)
    if any(getattr(marker,f)!=getattr(lock,f) for f in ("instance_id","lock_identity","pid","parent_pid")):raise ReadinessError("marker/lock mismatch")
    if (marker.owner_qq,marker.bot_qq)!=(owner,bot):raise ReadinessError("READY identity mismatch")
    if not pid_probe(marker.pid) and not pid_probe(marker.parent_pid):raise ReadinessError("READY process dead")
    return marker

class ReadinessCoordinator:
    def __init__(self,*,database_factory:Callable[[],Database],marker_path:str|Path,lock:InstanceLock,owner_qq:str,bot_qq:str,model_capability:ModelCapability,identity_provider:Callable[[],IdentityEvidence],ws_provider:Callable[[],WebSocketEvidence],memory_worker_provider:Callable[[],WorkerEvidence],initiative_worker_provider:Callable[[],WorkerEvidence],required_context_tokens:int=262144,clock:Callable[[],datetime]|None=None,build_id:str|None=None):
        self.database_factory=database_factory;self.marker_path=Path(marker_path);self.lock=lock;self.owner_qq=_decimal(owner_qq,"owner_qq");self.bot_qq=_decimal(bot_qq,"bot_qq")
        if self.owner_qq==self.bot_qq:raise ReadinessError("owner and bot must differ")
        if not isinstance(model_capability,ModelCapability):raise TypeError("model_capability invalid")
        if isinstance(required_context_tokens,bool) or not isinstance(required_context_tokens,int) or required_context_tokens<=0:raise ValueError("required_context_tokens must be a positive int")
        self.model_capability=model_capability;self.required_context_tokens=required_context_tokens;self.identity_provider=identity_provider;self.ws_provider=ws_provider;self.memory_worker_provider=memory_worker_provider;self.initiative_worker_provider=initiative_worker_provider;self.clock=clock or (lambda:datetime.now(timezone.utc));self.build_id=current_build_id() if build_id is None else _build_id(build_id,"build_id");self._marker:ReadyMarker|None=None
    def start(self)->ReadyMarker:
        self.model_capability.assert_supports(self.required_context_tokens)
        acquired_here = self.lock.record is None
        lock = self.lock.acquire() if acquired_here else self.lock.record
        if lock is None:
            raise ReadinessError("instance lock is unavailable")
        database:Database|None=None
        try:
            database=self.database_factory();_verify_database(database)
            identity=self.identity_provider()
            if not isinstance(identity,IdentityEvidence) or (identity.owner_qq,identity.bot_qq)!=(self.owner_qq,self.bot_qq):raise ReadinessError("identity mismatch")
            report=recover_database(database,conversation_id=self.owner_qq)
            ws=self.ws_provider()
            if not isinstance(ws,WebSocketEvidence) or (ws.authenticated_owner_qq,ws.authenticated_bot_qq)!=(self.owner_qq,self.bot_qq):raise ReadinessError("websocket mismatch")
            memory=self.memory_worker_provider();initiative=self.initiative_worker_provider()
            if not isinstance(memory,WorkerEvidence) or not isinstance(initiative,WorkerEvidence):raise ReadinessError("worker evidence type invalid")
            marker=ReadyMarker.from_dict({"schema":"qichi-ready","version":READY_MARKER_VERSION,"instance_id":lock.instance_id,"lock_identity":lock.lock_identity,"pid":lock.pid,"parent_pid":lock.parent_pid,"owner_qq":self.owner_qq,"bot_qq":self.bot_qq,"database_schema_version":SCHEMA_VERSION,"last_recovered_sequence":report.last_recovered_sequence,"ws_connection_id":ws.connection_id,"memory_worker_id":memory.worker_id,"initiative_worker_id":initiative.worker_id,"started_at_utc":self.clock().astimezone(timezone.utc).isoformat(),"provider":self.model_capability.provider,"model":self.model_capability.model_id,"context_window":self.required_context_tokens,"build_id":self.build_id})
            write_ready_marker(self.marker_path,marker,lock_path=self.lock.path);self._marker=marker;return marker
        except BaseException:
            self._remove_marker()
            if acquired_here:
                self.lock.release()
            raise
        finally:
            if database is not None:database.close()
    def _remove_marker(self)->bool:
        if self._marker is None:return False
        try:current=_read(self.marker_path,ReadyMarker.from_dict,"marker")
        except ReadinessError:self._marker=None;return False
        if current!=self._marker:self._marker=None;return False
        self.marker_path.unlink();self._marker=None;return True
    def stop(self)->None:self._remove_marker();self.lock.release()
