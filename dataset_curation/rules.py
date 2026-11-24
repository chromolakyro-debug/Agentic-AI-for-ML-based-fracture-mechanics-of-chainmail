"""Exact requested_style mapping authorized by the user; no visual decisions."""
CLASSES = [
    ('01', '写实点画', 'Realistic Stippling', 'realistic_stippling', 750),
    ('02', '写实笔绘', 'Realistic Ink Drawing', 'realistic_ink_drawing', 1500),
    ('03', '几何三角', 'Triangular Geometry', 'triangular_geometry', 750),
    ('04', '几何破碎', 'Fragmented Geometry', 'fragmented_geometry', 750),
    ('05', '像素线描', 'Pixel Line Art', 'pixel_line_art', 759),
    ('06', '像素实心', 'Solid Pixel Art', 'solid_pixel_art', 766),
    ('07', '卡通', 'Cartoon', 'cartoon', 1500),
    ('08', '剪影', 'Silhouette', 'silhouette', 1082),
]
SERIES = {
    'halftone': '01', 'binary_pen': '02', 'woodcut': '02',
    'triangular_lattice': '03', 'geometric': '04', 'outline_pixel': '05',
    'whiskered_sprite': '06', 'comic': '07', 'inverted_line': '07',
}
# This finite expansion is an exact dictionary lookup, NOT prefix/substring matching.
KEEP = {f'{series}__{suffix}': group for series, group in SERIES.items()
        for suffix in ('00', '01', '02', '03')}
KEEP.update({
    'portrait_pixel_tabby': '05',
    'pixel_fluffy': '06', 'pixel_tabby': '06', 'pixel_tuxedo': '06',
    'bold_ink': '08', 'solid_silhouette': '08', 'broad_folk': '08',
    'filled_rounded_ink': '08', 'filled_cartoon': '08', 'filled_silhouette': '08',
    'filled_angular_cutpaper': '08', 'rounded_cartoon': '08',
    'angular_graphic': '08', 'paper_cut': '08', 'soft_folk': '08',
})
DELETE = frozenset((
    'anatomical_ink', 'fur_woodcut', 'tabby_linocut', 'portrait_tabby_stencil',
    'pixel_round', 'tuxedo_graphic', 'portrait_tuxedo', 'hollow_ink',
    'angular_contour', 'contour_and_fill', 'clean_outline', 'sculpted_cutpaper',
    'portrait_fluffy_ink',
))
assert len(KEEP)==51 and len(DELETE)==13 and not set(KEEP)&DELETE
SEED = 20261116


def rules_document():
    return dict(classes=[dict(group_id=g,group_name_zh=zh,group_name_en=en,
                             group_label=label,expected_count=n) for g,zh,en,label,n in CLASSES],
                exact_requested_style_mapping={
                    **{k:dict(action='keep',group_id=v) for k,v in sorted(KEEP.items())},
                    **{k:dict(action='delete',reason='User-specified whole requested_style exclusion') for k in sorted(DELETE)}},
                unmatched_policy='STOP and report; never infer or delete unknown labels',
                sorting='Within each class: original numeric sample ID, then original relative path',
                new_identity='Class-local cat_000001 filename; globally unique new_id is 01_cat_000001, etc.',
                split=dict(mode='class_stratified_resplit',seed=SEED,ratios=[.8,.1,.1],
                    user_resolution='User explicitly chose re-stratification by the eight new classes over the conflicting filter-only acceptance clause.',
                    preserve_old_split='Archived verbatim and also filtered/reindexed in provenance; not the active split'),
                normalization='Retain original CFM mean/std exactly; label as frozen legacy statistics, not fitted to the new training split')
