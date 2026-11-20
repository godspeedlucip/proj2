"""Run independent RSTAR ablation groups on available GPUs and report progress."""
from datetime import datetime
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from rstar4d.utils.io import append_jsonl, atomic_json, read_json


ARMS = ('MPB','PB','MB','MP','B','M','P','baseline')
LAUNCH_ORDER = ('MPB','baseline','PB','MB','MP','B','M','P')
MIN_FREE_MIB = 36 * 1024
REPORT_SECONDS = 1800


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read_optional(path):
    return read_json(path) if path.is_file() else {}


def process_start(pid):
    stat = Path(f'/proc/{pid}/stat')
    try:
        fields = stat.read_text().rsplit(')', 1)[-1].split()
    except FileNotFoundError:
        return None
    return fields[19] if fields[0] != 'Z' else None


def latest_logged(path, event=None):
    if not path.exists():
        return {}
    with path.open('rb') as log:
        size = log.seek(0, 2)
        log.seek(max(0, size - 256 * 1024))
        for line in reversed(log.read().splitlines()):
            try:
                entry = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if event is None or entry.get('event') == event:
                return entry
    return {}


def devices():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'],text=True,timeout=15)
    pids = {}
    for line in processes.splitlines():
        uuid, pid = (field.strip() for field in line.split(','))
        pids.setdefault(uuid,[]).append(int(pid))
    cards = {}
    for line in output.splitlines():
        gpu, uuid, memory, activity = (field.strip() for field in line.split(','))
        cards[int(gpu)] = {'free_mib':int(memory),'utilization':int(activity),'compute_pids':pids.get(uuid,[])}
    return cards


def recoverable(job):
    if job['state']=='waiting' or (job['state']=='interrupted' and job.get('attempt',0)<2):
        return True
    if job['state']!='failed' or job.get('attempt',0)>=2:
        return False
    with Path(job['log']).open('rb') as log:
        size = log.seek(0,2)
        log.seek(max(0,size-8192))
        tail = log.read().lower()
    return any(message in tail for message in (b'cuda out of memory',b'cuda error: out of memory',
                                               b'resource temporarily unavailable',b'stale file handle'))


class Dispatch:
    def __init__(self, root):
        self.root = root
        self.logs = root/'logs/formal'
        self.data = root/'training_data'
        self.runs = root/'training_output/simulated'
        self.ablation = self.runs/'ablation'
        self.segmenter = self.runs/'mask-segmenter'
        self.manifest = self.data/'simulated/heart_main/annotated/manifest.jsonl'
        self.measured = self.data/'real/heart_only/manifest.jsonl'
        self.split = self.data/'split.json'
        self.best = self.runs/'shared-stage1/checkpoints/best.pt'
        self.test = root/'test_results/simulated'
        self.children = {}

    def summary(self, arm):
        return self.test/arm/'evaluation/summary.json'

    def publish_comparison(self, groups):
        audit = read_json(self.data/'data_audit.json')
        expected = (audit['scan_counts']['test'],audit['phase_counts']['test'])
        rows = []
        for arm, group in groups.items():
            metrics = group['evaluation'].get('metrics')
            if not metrics or group['evaluation']['state']!='complete':
                continue
            if (metrics['case_count'],metrics['phase_count'])!=expected:
                raise ValueError(f'{arm} does not cover the fixed test scans and phases')
            dice = metrics['dice']
            row = {'group':arm,'training_gpu':group['training']['gpu'],'best_epoch':metrics['stage2_best_epoch'],
                   'test_scans':expected[0],'test_phases':expected[1],
                   'heart_dice':dice['heart_outer']['scan_equal_mean'],'heart_dice_phases':dice['heart_outer']['phases'],
                   'lv_dice':dice['lv']['scan_equal_mean'],'lv_dice_phases':dice['lv']['phases']}
            for name, values in metrics['image'].items():
                row[name] = values['fov/prediction']['scan_equal_mean']
            rows.append(row)
        if not rows:
            return
        files = {row['group']:str(self.summary(row['group'])) for row in rows}
        atomic_json(self.test/'comparison.json',{'complete':len(rows)==len(ARMS),'groups':rows,'details':files})
        with tempfile.NamedTemporaryFile('w',encoding='utf-8-sig',newline='',dir=self.test,delete=False) as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        Path(file.name).replace(self.test/'comparison.csv')

    def tracked(self, name):
        job = read_optional(self.logs/f'{name}.job.json')
        if not job:
            return {'state':'waiting'}
        if name in self.children and self.children[name].poll() is not None:
            del self.children[name]
        exit_file = self.logs/f'{name}.exit.json'
        if exit_file.is_file():
            result = read_json(exit_file)
            return {**job, **result, 'state':'complete' if result['returncode'] == 0 else 'failed'}
        alive = process_start(job['pid']) == job['process_start']
        return {**job, 'state':'running' if alive else 'interrupted'}

    def external(self, name):
        pid_file = self.logs/f'{name}.pid'
        exit_file = self.logs/f'{name}.exit'
        if not pid_file.is_file():
            return {'state':'waiting'}
        pid = int(pid_file.read_text())
        gpu = self.logs/f'{name}.gpu'
        result = {'pid':pid, 'gpu':int(gpu.read_text()) if gpu.is_file() else None}
        if exit_file.is_file() and exit_file.read_text().strip():
            code = int(exit_file.read_text())
            return {**result, 'returncode':code, 'state':'complete' if code == 0 else 'failed'}
        return {**result, 'state':'running' if process_start(pid) else 'interrupted'}

    def launch(self, name, gpu, command, previous):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'PYTHONUNBUFFERED':'1',
               'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'2',
               'MPLCONFIGDIR':str(self.root/'preflight/mpl-cache')}
        target = self.logs/f'{name}.exit.json'
        log = self.logs/f'{name}.log'
        target.unlink(missing_ok=True)
        with log.open('a') as output:
            print(f'\n{now()} starting on physical GPU {gpu}\n{subprocess.list2cmdline(command)}',file=output,flush=True)
            worker = subprocess.Popen([sys.executable, '-m', 'experiments.formal_dispatch',
                                       '--worker', str(target), '--', *command],
                                      stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                      env=env, start_new_session=True)
        self.children[name] = worker
        job = {'gpu':gpu, 'pid':worker.pid, 'process_start':process_start(worker.pid),
               'attempt':previous.get('attempt',0)+1, 'started':now(), 'log':str(log)}
        atomic_json(self.logs/f'{name}.job.json',job)
        append_jsonl(self.logs/'launches.jsonl',{'task':name,**job})
        print(f'{job["started"]} {name} GPU {gpu} PID {worker.pid}',flush=True)

    def train_command(self, arm):
        run = self.ablation/arm
        command = [sys.executable,'-m','rstar4d','train','--stage','2',
                   '--config',str(self.root/f'configs/simulated/arms_parallel/{arm}.yaml'),
                   '--manifest',str(self.manifest),'--split',str(self.split),'--run-dir',str(run)]
        if (run/'run_identity.json').is_file():
            return [*command,'--resume',str(run/'checkpoints/last.pt')]
        return [*command,'--init-from-stage1',str(self.best)]

    def evaluation_command(self, arm, tile):
        return [sys.executable,'-m','experiments.formal_metrics',
                '--checkpoint',str(self.ablation/arm/'checkpoints/best.pt'),
                '--manifest',str(self.manifest),'--split',str(self.split),
                '--segmenter',str(self.segmenter),'--out',str(self.test/arm),
                '--measured',str(self.measured),'--measured-out',str(self.root/'test_results/real'/arm),
                '--tile',*map(str,tile['tile']),'--overlap',*map(str,tile['overlap'])]

    def state(self):
        stage1 = self.external('stage1')
        stage1['training'] = self.training_progress(self.runs/'shared-stage1',60)
        annotations = self.external('annotations')
        annotations['cached_cases'] = len(list(self.manifest.parent.glob('**/*.h5')))
        segmenter = self.tracked('segmenter')
        segmenter['training'] = latest_logged(self.segmenter/'train.jsonl')
        groups = {}
        for arm in ARMS:
            training = self.tracked(f'train-{arm}')
            training['training'] = self.training_progress(self.ablation/arm,30)
            evaluation = self.tracked(f'eval-{arm}')
            evaluation['completed_test_scans'] = len(list((self.test/arm/'evaluation').glob('*.json'))) - int(self.summary(arm).is_file())
            summary = read_optional(self.summary(arm))
            if summary:
                evaluation['metrics'] = {field:summary[field] for field in ('image','dice','case_count','phase_count','stage2_best_epoch')}
            groups[arm] = {'training':training,'evaluation':evaluation}
        return {'checked_at':now(),'stage1':stage1,'annotations':annotations,
                'segmenter':segmenter,'groups':groups}

    @staticmethod
    def training_progress(run, epochs):
        epoch = latest_logged(run/'train.jsonl','epoch')
        batch = latest_logged(run/'train.jsonl','batch')
        return {'epochs':epochs,'completed_epoch':epoch.get('epoch',0),
                'validation_l1':epoch.get('validation_l1'),'epoch_seconds':epoch.get('seconds'),
                'batch_epoch':batch.get('epoch'),'draw':batch.get('draw'),
                'physical_batch_size':batch.get('physical_batch_size'),
                'effective_batch_size':epoch.get('effective_batch_size'),
                'samples_by_physical_batch':epoch.get('samples_by_physical_batch'),
                'optimizer_step':batch.get('step')}

    def maybe_launch(self, state, cards):
        stage1, annotations, segmenter = (state[name] for name in ('stage1','annotations','segmenter'))
        stages = [stage1,segmenter]
        for group in state['groups'].values():
            stages.extend(group.values())
        occupied = {job.get('gpu') for job in stages if job['state']=='running'}
        free = {gpu for gpu,values in cards.items()
                if gpu not in occupied and not values['compute_pids']
                and values['free_mib'] >= MIN_FREE_MIB and values['utilization'] <= 10}
        used = {group['training'].get('gpu') for group in state['groups'].values()}

        def select(preferred=None, used=()):
            return min(free,key=lambda gpu: (gpu in used, gpu!=preferred, -cards[gpu]['free_mib'], gpu))

        def run(name, command, previous, gpu):
            self.launch(name,gpu,command,previous)
            free.remove(gpu)

        if annotations['state']!='complete' or not self.manifest.is_file():
            return
        if free and recoverable(segmenter):
            command = [sys.executable,'-m','experiments.frozen_cardiac_segmenter',
                       '--manifest',str(self.manifest),'--split',str(self.split),'--out',str(self.segmenter)]
            run('segmenter',command,segmenter,select(preferred=5))
        if stage1['state']!='complete' or not self.best.is_file():
            return
        for arm in LAUNCH_ORDER:
            previous = state['groups'][arm]['training']
            if free and recoverable(previous):
                gpu = select(preferred=previous.get('gpu'),used=used)
                run(f'train-{arm}',self.train_command(arm),previous,gpu)
                used.add(gpu)
        tile = read_optional(self.logs/'inference_preflight.json')
        if segmenter['state']!='complete' or not (self.segmenter/'frozen.json').is_file() or not tile:
            return
        for arm, group in state['groups'].items():
            previous = group['evaluation']
            if free and group['training']['state']=='complete' and recoverable(previous):
                run(f'eval-{arm}',self.evaluation_command(arm,tile),previous,select(group['training'].get('gpu')))

    def report(self, state):
        append_jsonl(self.logs/'progress_30min.jsonl',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)

    def loop(self):
        next_report = time.monotonic()
        started = self.logs/'stage1.started'
        if started.is_file() and (self.logs/'progress_30min.jsonl').is_file():
            origin = datetime.fromisoformat(started.read_text().strip()).timestamp()
            elapsed = max(0.,time.time()-origin)
            next_report += REPORT_SECONDS-(elapsed%REPORT_SECONDS)
        while True:
            state = self.state()
            try:
                cards = devices()
            except (subprocess.SubprocessError, ValueError) as exc:
                state['gpu_query_error'] = str(exc)
                cards = {}
            state['gpus'] = cards
            atomic_json(self.logs/'progress.json',state)
            groups = state['groups']
            complete = all(g['evaluation']['state']=='complete' and 'metrics' in g['evaluation'] for g in groups.values())
            if complete or time.monotonic() >= next_report:
                self.publish_comparison(groups)
                self.report(state)
                next_report = time.monotonic() + REPORT_SECONDS
            if complete:
                return 0
            self.maybe_launch(state,cards)
            time.sleep(min(45,max(1,next_report-time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--root',type=Path)
    mode.add_argument('--worker',type=Path)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker:
        command = args.command[1:] if args.command[:1]==['--'] else args.command
        code = subprocess.call(command)
        atomic_json(args.worker,{'returncode':code,'finished':now()})
        return code
    dispatch = Dispatch(args.root.resolve())
    with (dispatch.logs/'dispatcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(dispatch.logs/'dispatcher.pid.json',{'pid':os.getpid(),'started':now()})
        return dispatch.loop()


if __name__ == '__main__':
    raise SystemExit(main())
