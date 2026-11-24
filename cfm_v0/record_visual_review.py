"""Record the actually completed fixed-40 pixel inspection; does not perform automatic vision."""
from datetime import datetime, timezone
from common import OUT, read_json, save_json, sha_file

JUDGMENTS=[
 [('R','完整猫形 silhouette，可见头耳、尾和腿；有小悬浮足状块'),('D','pixel cat 面部可辨，但耳、身体与脚在边缘裁切')],
 [('R','猫形 silhouette 连贯；比例偏粗并有小足状伪影'),('D','pixel cat 面部可辨，头身与脚的边界裁切')],
 [('R','猫形 silhouette 连贯，尾与头耳明确；局部足部粗糙'),('D','pixel cat 面部可辨，耳、脚和背部形态不完整')],
 [('R','猫形 silhouette 与尾连贯；局部足状孤立块'),('D','pixel cat 面部可辨，主体触边且四肢不完整')],
 [('R','细长坐立猫，头耳、脸和身体连贯'),('R','简化坐猫，耳、眼和身体连贯，轮廓粗糙')],
 [('D','猫脸明确，但黑色背景中身体轮廓缺失，脚与身体关系异常'),('R','坐猫的耳、脸、前腿可辨且主体连贯；存在少量孤立细线')],
 [('D','猫轮廓被拉长，头部/躯干轮廓缺损且触边'),('R','纹理猫的头、身体、四肢与尾可辨')],
 [('R','横向 lattice 猫，头、长躯干与前腿可辨，构图很长'),('D','低伏四足形有猫特征，但脸与身体局部熔合、失真')],
 [('R','细长站立猫的头、双耳与前腿连贯'),('R','小型坐猫，头、耳、身体与脚连贯')],
 [('D','pixel 猫耳脸可辨，但身体轮廓缺失、腿和尾的线条断续'),('R','小型坐猫，头耳与前腿清楚，略粗糙')],
 [('U','稀疏白色线条与下方碎片，无法稳定辨认完整猫头和身体'),('D','猫耳/头的局部特征存在，但头身与肢体高度扭曲')],
 [('D','横向猫轮廓可辨，但极端拉长且身体/腿触边裁切'),('D','猫的局部特征存在，头身及腿部断裂融合')],
 [('R','小型线描猫，头耳、身躯、脚与尾可辨'),('D','pixel 猫的头耳可辨，但身体描边缺失、尾腿线条脱离')],
 [('R','连贯的黑猫 silhouette，举爪、头耳和尾可辨'),('D','pixel 猫脸可辨，但头身大幅裁切且肢体不完整')],
 [('R','清晰坐猫，脸、双耳、身体、前脚及尾连贯'),('R','完整侧身猫，脸、耳、四肢及尾连贯')],
 [('R','小型细长站猫，头耳与前腿连贯'),('R','小型坐猫，双耳、脸及前脚可辨')],
 [('D','横向猫可辨，但极端拉长，头、尾和脚被边界裁切'),('R','完整侧身白猫的头耳、四腿及尾可辨；左下有孤立块')],
 [('D','横向 lattice 猫可辨，但尾头触边裁切、身体过长'),('R','侧身白猫，脸耳和躯干四肢连贯；左下有孤立块')],
 [('D','横向 lattice 猫可辨，但头脚触边且躯干拉长'),('D','粗壮四足形，有弱猫特征，但头耳不清、体型和腿部失真')],
 [('D','低矮 lattice 猫局部可辨，但严重拉长且头脚触边裁切'),('D','猫耳/头的局部特征可辨，但头身和肢体碎裂融合')]
]


def main():
    assert sha_file(OUT/'final'/'manifest.json')=='b12fbda5b21c7263b4520f83b7b4e731cda45f64670d1e82061675e1ba91001a', \
        'These judgments apply only to the actually inspected original manifest; new outputs require new visual inspection.'
    manifest=read_json(OUT/'final'/'manifest.json')
    categories={'R':'recognizable_cat','D':'cat_like_but_distorted','U':'unrecognizable'}
    rows=[]
    for sample in manifest['samples']:
        if sample['seed'] not in manifest['seeds'][:2]:continue
        category,note=JUDGMENTS[int(sample['target_id'][1:])][manifest['seeds'].index(sample['seed'])]
        rows.append(dict(key=sample['key'],target_id=sample['target_id'],seed=sample['seed'],
                         sha256_pixels=sample['sha256_pixels'],category=categories[category],note=note))
    save_json(OUT/'final'/'visual_inspection.json',dict(reviewer='assistant actual pixel inspection; not human confirmation',
             recorded_at_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha_file(OUT/'final'/'manifest.json'),
             scope='first two pre-frozen seeds at all 20 targets, 40 images; no FEM selection',
             evidence=['visual_review_fixed_seeds.png','visual_details/T10_seed730000.png','visual_details/T10_seed730001.png',
                       'visual_details/T18_seed730001.png','visual_details/T19_seed730001.png'],
             summary='高刚度正 a 及部分轮廓风格出现拉长、触边或碎裂；同一 noise seed 在不同条件下仍保留明显的姿态/风格关联。基本连贯不等于没有伪影或具备制造可行性。',
             samples=rows))
    print('Recorded actual visual judgments for',len(rows),'images')


if __name__=='__main__':main()
