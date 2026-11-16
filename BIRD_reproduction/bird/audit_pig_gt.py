"""Compare an existing GT metadata file with the new measurement plan; no images read."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import file_sha256


def audit(plan_path,gt_config_path,out):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    gt=json.loads(Path(gt_config_path).read_text(encoding='utf-8'))
    differences={k:dict(current_plan=v,existing_gt=gt.get('settings',{}).get(k))
                 for k,v in plan['settings'].items() if v!=gt.get('settings',{}).get(k)}
    intensity=gt.get('intensity_processing')
    raw=intensity==plan['intensity_processing'] and gt.get('calibration_csv') is None
    result=dict(plan_sha256=file_sha256(plan_path),gt_config_path=str(Path(gt_config_path).resolve()),
                gt_config_sha256=file_sha256(gt_config_path),settings_differences=differences,
                existing_gt_intensity=intensity,measurement_intensity=plan['intensity_processing'],
                compatible_raw_mu_evaluation=not differences and raw,
                scope='Metadata only; masks, image alignment and quality still require separate validation.',
                recommendation='Use matching settings and raw-mu GT from the current generator, or explicitly document and align the calibration domain.')
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--gt-config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();audit(args.plan,args.gt_config,args.out)
