"""
Extract the pre-class layer features and cluster them with KMeans.
Save the per-pixel assigned clusters with a color.

Note, this requires sklearn and matplotlib.
"""
import pdb

import torch
import semask
import torchvision
from torchvision import io
from sklearn.cluster import KMeans
from matplotlib import cm as colormap


def main():
    # Parameters
    n_cluster = 10
    n_kmeans_sample = 10000
    device = "cpu"
    img_name = "terrain.png"
    model_name = "ade20k_base"

    # Load image
    img = io.read_image(img_name).to(torch.float32) / 255
    img = torchvision.transforms.functional.normalize(
        img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    img = img.to(device).unsqueeze(0)

    # Load model and get features
    model = semask.load_model(model_name).eval()
    model.to(device)
    with torch.no_grad():
        features = model.inference_features(img, rescale=True)

    # Cluster with a subset of the features
    features = features.view(-1, img.shape[2] * img.shape[3])
    features = features.permute((1, 0))
    kmeans_obj = KMeans(n_clusters=n_cluster)
    kmeans_feat = features[torch.randperm(features.shape[0])[:n_kmeans_sample]]
    kmeans_obj.fit(kmeans_feat)

    # Visualize cluster assignments
    cluster_idx = kmeans_obj.predict(features)
    colors = colormap.get_cmap("tab20")
    colors = torch.tensor([colors(i)[:3] for i in range(n_cluster)])
    viz_img = colors[cluster_idx].permute((1, 0)).view(-1, img.shape[2], img.shape[3])
    torchvision.utils.save_image(viz_img, f"clustered_image.png")


if __name__ == "__main__":
    main()
