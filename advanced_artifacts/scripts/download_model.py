"""Download a revision-pinned public checkpoint through a reachable HF mirror."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
HOST = "https://hf-mirror.com"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEST = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_WORKERS = 4


def download_large_file(url, partial, session):
    with session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                     timeout=(15, 30)) as probe:
        if probe.status_code != 206 or "Content-Range" not in probe.headers:
            raise RuntimeError("The model host does not support byte-range downloads")
        total = int(probe.headers["Content-Range"].rsplit("/", 1)[1])

    parts_dir = partial.parent / f".{partial.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, total, CHUNK_SIZE)):
        end = min(start + CHUNK_SIZE, total) - 1
        part = parts_dir / f"{index:04d}.part"
        size = end - start + 1
        if not part.exists() or part.stat().st_size != size:
            chunks.append((index, start, end, part))

    def fetch_chunk(index, start, end, part):
        expected = end - start + 1
        temporary = part.with_suffix(".partial")
        for attempt in range(6):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{total}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError(f"Unexpected range response for chunk {index}")
                    with temporary.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                handle.write(block)
                if temporary.stat().st_size != expected:
                    raise RuntimeError(f"Incomplete range response for chunk {index}")
                temporary.replace(part)
                return expected
            except (requests.RequestException, OSError, RuntimeError):
                temporary.unlink(missing_ok=True)
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))

    ready = total - sum(end - start + 1 for _, start, end, _ in chunks)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        for offset in range(0, len(chunks), DOWNLOAD_WORKERS):
            batch = chunks[offset:offset + DOWNLOAD_WORKERS]
            futures = [executor.submit(fetch_chunk, *chunk) for chunk in batch]
            for future in as_completed(futures):
                ready += future.result()
                print(partial.name, round(ready / 2**20), "/", round(total / 2**20), "MiB", flush=True)

    with partial.open("wb") as output:
        for index in range((total + CHUNK_SIZE - 1) // CHUNK_SIZE):
            part = parts_dir / f"{index:04d}.part"
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if partial.stat().st_size != total:
        raise RuntimeError("Assembled model file has an unexpected size")
    shutil.rmtree(parts_dir)
    return total


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(f"{HOST}/api/models/{MODEL}/revision/{REVISION}", timeout=(10, 30))
    response.raise_for_status()
    info = response.json()
    revision = info["sha"]
    if revision != REVISION:
        raise RuntimeError(f"Unexpected model revision: {revision}")
    wanted = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors",
    }
    files = [x["rfilename"] for x in info["siblings"] if x["rfilename"] in wanted]
    manifest = {"model_id": MODEL, "revision": revision, "download_host": HOST, "files": {}}
    print("revision", revision, flush=True)
    for name in files:
        target = DEST / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            start = time.perf_counter()
            last = start
            url = f"{HOST}/{MODEL}/resolve/{revision}/{name}"
            if name == "model.safetensors":
                size = download_large_file(url, partial, session)
            else:
                with session.get(url, stream=True, timeout=(15, 45)) as download:
                    download.raise_for_status()
                    with partial.open("wb") as handle:
                        size = 0
                        for block in download.iter_content(1024 * 1024):
                            if block:
                                handle.write(block)
                                size += len(block)
                                now = time.perf_counter()
                                if now - last >= 15:
                                    print(name, round(size / 2**20, 1), "MiB", flush=True)
                                    last = now
            partial.replace(target)
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        manifest["files"][name] = {"bytes": target.stat().st_size, "sha256": digest.hexdigest()}
        print("ready", name, target.stat().st_size, flush=True)
        (DEST / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
