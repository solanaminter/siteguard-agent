"""Storage abstraction: local filesystem or S3 (boto3).

The local backend mirrors the S3 key layout so the container runs
identically on a laptop and on ECS; only STORAGE_BACKEND changes.
"""
import os


class Storage:
    def put(self, key: str, local_path: str) -> str:
        raise NotImplementedError

    def url(self, key: str) -> str:
        raise NotImplementedError


class LocalStorage(Storage):
    def __init__(self, root: str = "assets/evidence"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, key: str, local_path: str) -> str:
        import shutil
        dest = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local_path, dest)
        return dest

    def url(self, key: str) -> str:
        return os.path.join(self.root, key)


class S3Storage(Storage):
    def __init__(self, bucket: str, region: str = "us-west-2"):
        import boto3
        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)

    def put(self, key: str, local_path: str) -> str:
        self.s3.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def url(self, key: str) -> str:
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=3600
        )


def get_storage(backend: str = "local", bucket: str = "siteguard-evidence",
                 region: str = "us-west-2") -> Storage:
    if backend == "s3":
        try:
            return S3Storage(bucket, region)
        except Exception as e:  # no creds in dev -> fall back, loudly
            print(f"[storage] S3 unavailable ({e}); falling back to local")
    return LocalStorage()
