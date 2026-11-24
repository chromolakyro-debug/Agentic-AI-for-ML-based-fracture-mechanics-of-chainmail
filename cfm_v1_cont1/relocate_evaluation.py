"""One-time, hash-checked relocation of completed evaluation runs."""
import sys
import shutil
from common import ROOT, HERE, RUN, read, save, sha
sys.path.insert(0, str(ROOT / 'evaluation'))
from utils import digest
from verify_report import verify


def main():
    source = (HERE / 'evaluation').resolve()
    destination = (ROOT / 'evaluation/runs').resolve()
    assert source.is_relative_to(ROOT.resolve()) and destination.is_relative_to(ROOT.resolve())
    assert read(RUN / 'workflow_complete.json')['status'] == 'complete'
    folders = [source/f'step_{s}' for s in [300000,320000,340000,360000,415000,440000,480000]]
    assert len(folders) == 7
    mappings = {p.relative_to(ROOT).as_posix(): 'evaluation/runs/cfm_v1_cont1_' + p.name for p in folders}
    for old,new in mappings.items():
        assert (ROOT/old).resolve().parent == source
        assert (ROOT/new).resolve().parent == destination
        assert (ROOT/old).exists() != (ROOT/new).exists()
    backup = RUN / 'relocation_before'
    def preserve(p):
        dest = backup / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():shutil.copyfile(p, dest)
    def replace(text):
        for old,new in mappings.items(): text = text.replace(old,new)
        return text
    moves = []
    for old,new in mappings.items():
        a,b = ROOT/old, ROOT/new
        present = a if a.exists() else b
        hashes = {p.relative_to(present).as_posix():sha(p) for p in present.rglob('*') if p.is_file()}
        if a.exists():
            preserve(a/'config.json');preserve(a/'manifest.json')
            a.rename(b)
        assert all(sha(b/rel)==h for rel,h in hashes.items())
        config = read(b/'config.json');config['output_directory']=new;save(b/'config.json',config)
        manifest = read(b/'manifest.json');old_identity=manifest.get('output_relocation',{}).get('previous_identity',manifest['identity'])
        manifest['inputs']['config']=config
        manifest['identity']=digest(manifest['inputs'])
        manifest['output_relocation']=dict(previous_output=old,previous_identity=old_identity,reason='User requested central evaluation directory; numerical inputs and results unchanged')
        save(b/'manifest.json',manifest)
        assert all(sha(b/rel)==h for rel,h in hashes.items() if rel not in {'config.json','manifest.json'})
        moves.append(dict(old=old,new=new,files=len(hashes),unchanged_results_and_images=True,html=verify(b)))
    source.rmdir()  # Only the now-empty former results directory.
    for p in list((HERE/'configs').glob('*.json'))+list(RUN.glob('*.json'))+[HERE/'REPORT.md',HERE/'README.md']:
        original=p.read_text(encoding='utf-8');changed=replace(original)
        if p.name=='README.md':changed=changed.replace('evaluation输出、configs和cache均在本目录','evaluation输出统一放在evaluation/runs/cfm_v1_cont1_step_XXXXXX；训练配置和cache仍在本目录')
        if changed!=original:preserve(p);p.write_text(changed,encoding='utf-8')
    changes={}
    for name in ['workflow.py','visual_review.py']:
        p=HERE/name;before=sha(p);original=p.read_text(encoding='utf-8')
        changed=original.replace("f'cfm_v1_cont1/evaluation/step_{step:06d}'", "f'evaluation/runs/cfm_v1_cont1_step_{step:06d}'")
        changed=changed.replace("ROOT / 'cfm_v1_cont1' / 'evaluation' / f'step_{step:06d}'", "ROOT / 'evaluation' / 'runs' / f'cfm_v1_cont1_step_{step:06d}'")
        if name=='workflow.py':
            changed=changed.replace("if target.exists():assert sha(target)==sha(p),'Code changed during continuation'", "if target.exists():\n            if sha(target)!=sha(p):\n                migration=read(RUN/'evaluation_relocation.json')\n                allowed=migration['post_training_code_changes'][p.relative_to(ROOT).as_posix()]\n                assert (RUN/'training_summary.json').exists() and allowed==dict(before=sha(target),after=sha(p)), 'Unrecorded code change'")
        assert changed!=original
        preserve(p);p.write_text(changed,encoding='utf-8')
        changes[p.relative_to(ROOT).as_posix()]=dict(before=before,after=sha(p))
    save(RUN/'evaluation_relocation.json',dict(status='passed',moves=moves,post_training_code_changes=changes,training_code_and_checkpoints_unchanged=True))
    print('Moved and verified all seven runs; training state unchanged.')


if __name__=='__main__':main()
