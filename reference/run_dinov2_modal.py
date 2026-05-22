"""Run DINOv2 embedding on Modal GPU for Mapillary or video frame images.

Usage:
    # Embed Mapillary index (5k images)
    modal run run_dinov2_modal.py --input gs://cs231n-navlm-data/mapillary/mapillary/zurich/images/ --output embeddings_mly.npz

    # Or embed local images uploaded to a Modal volume
    modal run run_dinov2_modal.py
"""

import modal

app = modal.App("navlm-dinov2")

# Build image with PyTorch + DINOv2 deps
dinov2_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "torchvision", "numpy", "Pillow", "tqdm")
)

# Create a persistent volume for storing images and results
vol = modal.Volume.from_name("navlm-data", create_if_missing=True)

@app.function(
    image=dinov2_image,
    gpu="A10G",  # 24GB, plenty for DINOv2 ViT-L/14 (~1.2GB)
    timeout=3600,
    volumes={"/data": vol},
)
def embed_images(image_dir: str = "/data/images", output_path: str = "/data/embeddings.npz"):
    """Compute DINOv2 ViT-L/14 embeddings for all JPGs in a directory."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    # Load DINOv2
    print("Loading DINOv2 ViT-L/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model = model.cuda().eval()

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Find all images
    img_dir = Path(image_dir)
    paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Found {len(paths)} images in {image_dir}")

    if len(paths) == 0:
        print("No images found! Check the path.")
        return

    ids = []
    embs = []

    # Process in batches
    batch_size = 32
    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i+batch_size]
        batch_tensors = []
        batch_ids = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))
                batch_ids.append(p.stem)
            except Exception as e:
                print(f"  Skip {p.name}: {e}")

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).cuda()
        with torch.no_grad():
            features = model(batch)  # (B, 1024) for ViT-L/14... actually 768 for some versions

        embs.append(features.cpu().numpy())
        ids.extend(batch_ids)

    embs = np.concatenate(embs, axis=0)
    ids = np.array(ids)

    print(f"Computed {len(ids)} embeddings, shape: {embs.shape}")

    # Save
    np.savez(output_path, ids=ids, embs=embs)
    print(f"Saved to {output_path}")

    # Commit volume so results persist
    vol.commit()

    return f"Done: {len(ids)} embeddings saved to {output_path}"


@app.function(
    image=dinov2_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/data": vol},
)
def embed_from_gcs(gcs_path: str, output_name: str = "embeddings.npz"):
    """Download images from GCS bucket, embed, save to volume."""
    import subprocess
    import os

    # Install gcloud
    subprocess.run(["pip", "install", "google-cloud-storage"], check=True)

    from google.cloud import storage
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm

    # Download from GCS
    local_dir = "/tmp/images"
    os.makedirs(local_dir, exist_ok=True)

    # Parse bucket and prefix
    # e.g. gs://cs231n-navlm-data/mapillary/mapillary/zurich/images/
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    print(f"Downloading from gs://{bucket_name}/{prefix}...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"  Found {len(blobs)} objects")

    for blob in tqdm(blobs):
        if blob.name.endswith(('.jpg', '.png')):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)

    # Now embed
    return embed_images.local(local_dir, f"/data/{output_name}")


@app.local_entrypoint()
def main():
    """Quick test: embed whatever is in the volume at /data/images/"""
    result = embed_images.remote()
    print(result)
