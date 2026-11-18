import argparse
import os
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset


def download_single_image(img_name: str, output_dir: Path) -> bool:
    """Download a single COCO image by its ID.

    Args:
        img_name: Raw image file name (e.g. '526756.jpg').
        output_dir: Local destination directory.

    Returns:
        bool: True if downloaded or already exists, False if failed.
    """
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    coco_12digit = f"{int(base_name):012d}.jpg" if base_name.isdigit() else img_name
    dest_path = output_dir / coco_12digit

    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True

    coco_urls = [
        f"http://images.cocodataset.org/train2017/{coco_12digit}",
        f"http://images.cocodataset.org/val2017/{coco_12digit}",
        f"http://images.cocodataset.org/val2014/COCO_val2014_{coco_12digit}",
        f"http://images.cocodataset.org/train2014/COCO_train2014_{coco_12digit}",
    ]

    for url in coco_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = resp.read()
                    with open(dest_path, "wb") as f:
                        f.write(data)
                    return True
        except Exception:
            continue

    return False


def main():
    parser = argparse.ArgumentParser(description="Download MS COCO images for ViVQA dataset")
    parser.add_argument("--output-dir", type=str, default="./data/images", help="Target folder for images")
    parser.add_argument("--cache-dir", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to download")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download threads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Downloader] Scanning ViVQA dataset for image references...")
    train_ds = load_dataset("ThucPD/ViVQA", split="train", cache_dir=args.cache_dir)
    test_ds = load_dataset("ThucPD/ViVQA", split="test", cache_dir=args.cache_dir)

    train_imgs = list(train_ds["image"])
    test_imgs = list(test_ds["image"])
    all_images = sorted(list(set(train_imgs + test_imgs)))
    if args.max_images:
        all_images = all_images[:args.max_images]

    print(f"[Downloader] Found {len(all_images)} unique images to verify/download.")
    print(f"[Downloader] Target directory: {output_dir.resolve()}")
    print(f"[Downloader] Parallel threads: {args.workers}")

    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_image, img, output_dir): img for img in all_images}
        pbar = tqdm(as_completed(futures), total=len(all_images), desc="Downloading images", unit="img")
        for fut in pbar:
            if fut.result():
                success_count += 1
            pbar.set_postfix({"downloaded": success_count})

    print(f"\n[Downloader] Complete: {success_count}/{len(all_images)} images ready in {output_dir}.")


if __name__ == "__main__":
    main()
