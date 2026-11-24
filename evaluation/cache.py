from utils import read,save,digest,sha

def get(path,identity):
    if not path.exists():return None
    try:
        item=read(path)
        if item['identity']!=identity or item['payload_sha256']!=digest(item['payload']):return None
        return item['payload']
    except (ValueError,KeyError,OSError):return None

def put(path,identity,payload):save(path,dict(identity=identity,payload_sha256=digest(payload),payload=payload))

def valid_array(path,meta_path,identity):
    meta=get(meta_path,identity)
    return meta if meta and path.exists() and sha(path)==meta['file_sha256'] else None
