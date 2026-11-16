import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import yaml
import torch
from rstar4d.cli import main, parser
from rstar4d.utils.io import read_json, read_jsonl, atomic_jsonl


class CliTests(unittest.TestCase):
    def test_all_commands_exposed(self):
        text=parser().format_help()
        for name in ['doctor','inventory-pig','validate-settings','make-phantom','simulate-pig','import-paired',
            'validate-data','convert-weights','train','pipeline','infer','evaluate','benchmark']:
            self.assertIn(name,text)

    @unittest.skipUnless(torch.cuda.is_available(),'End-to-end fixture uses actual TIGRE CUDA')
    def test_two_entry_points_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()): status=main(list(map(str,args)))
                self.assertEqual(status,0,args)
            run('make-phantom','--out',root/'raw','--views',200)
            run('simulate-pig','--manifest',root/'raw/raw_manifest.jsonl','--config',root/'raw/simulation.yaml','--out',root/'prepared')
            cfg={'device':'cuda','threads':2,'amp':True,'epochs':{'1':1,'2':1},'steps_per_epoch':{'1':8,'2':8},
                'stage1_patch':[1,32,32],'stage2_patches':[[8,32,32]],'log_every':1}
            (root/'train.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
            for entry in ['prepared','imported']:
                if entry=='imported':
                    rows=read_jsonl(root/'prepared/manifest.jsonl')
                    for r in rows:
                        r['paired_h5']=str(root/'prepared'/r['path'])
                        if r.get('projections'): r['projections']=str(root/'prepared'/r['projections'])
                    atomic_jsonl(root/'sources.jsonl',rows)
                    run('import-paired','--manifest',root/'sources.jsonl','--out',root/'imported')
                run('validate-data','--manifest',root/entry/'manifest.jsonl','--split',root/'raw/split.json')
                run('pipeline','--config',root/'train.yaml','--manifest',root/entry/'manifest.jsonl','--split',root/'raw/split.json','--run-dir',root/('run_'+entry))
                checkpoint=root/('run_'+entry)/'stage2/checkpoints/best.pt'
                run('infer','--checkpoint',checkpoint,'--manifest',root/entry/'manifest.jsonl','--out',root/('pred_'+entry),
                    '--tile',8,32,32,'--overlap',4,16,16,'--export','tif','nii')
                run('evaluate','--predictions',root/('pred_'+entry),'--manifest',root/entry/'manifest.jsonl','--out',root/('eval_'+entry))
                self.assertEqual(read_json(root/('eval_'+entry)/'summary.json')['cases'],1)

if __name__=='__main__': unittest.main()
