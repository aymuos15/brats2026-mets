import os
os.chdir(os.path.expanduser("~/brats2025/brats_eval"))
from panoptica import Panoptica_Evaluator

ev = Panoptica_Evaluator.load_from_config("brats_evaluation/configs/config_mets.yaml")
r = ev.evaluate("example/sample_data/pred/BraTS-MET-00630-001.nii.gz",
                "example/sample_data/ref/BraTS-MET-00630-001-seg.nii.gz")
for grp, res in r.items():
    d = res.to_dict(True)
    print(f"{grp:>4}: tp={d.get('tp')} fp={d.get('fp')} fn={d.get('fn')} rq={d.get('rq')} "
          f"n_ref={d.get('n_ref_instances')}")
    ri = d.get("reference_instances", [])
    if ri:
        print(f"      ref_instance keys: {sorted(ri[0].keys())}")
        for x in ri[:3]:
            print(f"      volume={x.get('volume')} is_matched={x.get('is_matched')} sq_dsc={x.get('sq_dsc')}")
print("SMOKE_OK")
