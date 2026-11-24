"""Seal the validated data release; no source images or numeric data are changed."""
from datetime import datetime, timezone
import shutil
from curate import OUT, HERE, read, save, sha


def main():
    assert read(OUT/'validation.json')['status'] == 'passed'
    assert read(OUT/'loader_validation.json')['status'] == 'passed'
    classes = read(OUT/'classes.json')['classes']
    expected_dirs = {c['group_id']+'_'+c['group_label'] for c in classes}
    assert {p.name for p in (OUT/'images').iterdir()} == expected_dirs
    assert len(list((OUT/'images').rglob('*.png'))) == 7857
    assert not list(OUT.rglob('*.tmp')) and not list(OUT.rglob('*.pyc'))
    assert (OUT/'REPORT.md').is_file() and (OUT/'README.md').is_file()
    assert len(read(OUT/'provenance'/'figure_samples.json')['samples']) == 32
    save(OUT/'provenance'/'figure_review.json', dict(
        image='eight_classes.png', reviewed_with='view_image',
        outcome='All eight class headings, counts and 32 sample labels are legible; no layout clipping observed.',
        scope='Layout inspection only; no images were reclassified or excluded based on appearance.'))
    for p in HERE.glob('*.py'):
        shutil.copy2(p, OUT/'provenance'/'curation_code'/p.name)
    commands = read(OUT/'provenance'/'execution_commands.json')
    command = 'dataset_curation/seal.py'
    if command not in commands['commands']:
        commands['commands'].append(command)
    save(OUT/'provenance'/'execution_commands.json', commands)
    excluded = {'integrity_manifest.json', 'STATUS.json'}
    files = [dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.relative_to(OUT).as_posix() not in excluded]
    save(OUT/'integrity_manifest.json', dict(
        algorithm='SHA256', excluded=sorted(excluded), files=files,
        file_count=len(files), total_bytes=sum(f['bytes'] for f in files)))
    # Re-read the serialized inventory and verify its contents before marking complete.
    saved = read(OUT/'integrity_manifest.json')
    for entry in saved['files']:
        path = OUT/entry['path']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    save(OUT/'STATUS.json', dict(
        status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        ready_for_data_loading=True, full_validation='validation.json',
        existing_loader_validation='loader_validation.json',
        integrity_manifest='integrity_manifest.json',
        integrity_manifest_sha256=sha(OUT/'integrity_manifest.json'),
        retained=7857, excluded=143, class_count=8,
        split_policy='User-confirmed new class-stratified 80/10/10; old splits archived.',
        training_note='This is a data release. Original normalization is preserved; future training sampler/support configuration belongs to a separate experiment.'))
    print(f'COMPLETE: {OUT}\nVerified {len(files)} file hashes; retained=7857; excluded=143; classes=8.')


if __name__ == '__main__':
    main()
