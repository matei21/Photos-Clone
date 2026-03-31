import os
import random
import cv2
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from glob import glob
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

#directml
try:
    import torch_directml

    HAS_DML = True
except ImportError:
    HAS_DML = False


# nns2
class TinyFaceNet(nn.Module):
    def __init__(self):
        super(TinyFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 9 * 7, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


#triplet
class FaceTripletDataset(Dataset):
    def __init__(self, dataset_path, num_triplets=2000):
        self.dataset_path = dataset_path
        self.num_triplets = num_triplets
        self.person_to_images = {}

        all_images = []
        for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
            all_images.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))

        for img_path in all_images:
            person = os.path.basename(os.path.dirname(img_path))
            if person not in self.person_to_images:
                self.person_to_images[person] = []
            self.person_to_images[person].append(img_path)

        self.valid_persons = [p for p in self.person_to_images.keys() if len(self.person_to_images[p]) >= 2]

        if len(self.valid_persons) < 2:
            raise ValueError("Dataset needs at least two different people, and some must have 2+ photos.")

    def __len__(self):
        return self.num_triplets

    def process_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (116, 140))
        img_tensor = torch.tensor(img).float().permute(2, 0, 1) / 255.0
        return img_tensor

    def __getitem__(self, idx):
        person_a = random.choice(self.valid_persons)
        anc_img_path, pos_img_path = random.sample(self.person_to_images[person_a], 2)

        person_b = random.choice(self.valid_persons)
        while person_b == person_a:
            person_b = random.choice(self.valid_persons)
        neg_img_path = random.choice(self.person_to_images[person_b])

        anc_tensor = self.process_image(anc_img_path)
        pos_tensor = self.process_image(pos_img_path)
        neg_tensor = self.process_image(neg_img_path)

        return anc_tensor, pos_tensor, neg_tensor


#training
def train_model(model, device, dataset_path, epochs=50, batch_size=128):
    print("\nSTARTING TRIPLET TRAINING PHASE")
    model.train()

    dataset = FaceTripletDataset(dataset_path, num_triplets=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    #optimizer
    optimizer = optim.Adagrad(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0

        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for anc, pos, neg in batch_iter:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()

            anc_embed = model(anc)
            pos_embed = model(pos)
            neg_embed = model(neg)

            loss = criterion(anc_embed, pos_embed, neg_embed)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_iter.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} Average Loss: {running_loss / len(dataloader):.4f}")

    print("Saving trained weights to facenet_trained.pth")
    torch.save(model.state_dict(), "facenet_trained.pth")
    print("Training complete!")


#qdrant
def setup_qdrant():
    client = QdrantClient(path="./face_cluster_db")
    collection_name = "casia_faces"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.EUCLID)
        )
    return client, collection_name


#embedding
def embed_dataset(model, client, collection_name, dataset_path="dataset", max_images=5000):
    print(f"\nSTARTING EMBEDDING PHASE")
    model.eval()
    device = next(model.parameters()).device

    image_paths = []
    for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
        image_paths.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))

    image_paths = sorted(image_paths)[:max_images]

    if not image_paths:
        print(f" 0 images found. Is your folder named exactly '{dataset_path}'?")
        return

    print(f"Embedding {len(image_paths)} images")

    for img_path in tqdm(image_paths, desc="Processing", unit="img"):
        true_label = os.path.basename(os.path.dirname(img_path))

        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (116, 140))

        img_tensor = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            vector_128d = model(img_tensor)[0].cpu().tolist()

        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_128d,
                payload={"true_label": true_label, "path": img_path}
            )]
        )


#clustering
def evaluate_clusters(client, collection_name):
    print("\nSTARTING CLUSTERING PHASE")
    results = client.scroll(collection_name=collection_name, limit=5000, with_vectors=True)[0]

    if not results:
        print("No data in DB.")
        return

    embeddings = np.array([p.vector for p in results])
    true_labels = [p.payload['true_label'] for p in results]

    print(f"Running Agglomerative Clustering on {len(embeddings)} faces...")

    #threshold pentru clustere - 0.2
    cluster_model = AgglomerativeClustering(n_clusters=None, metric='euclidean',
                                            linkage='average', distance_threshold=0.2)
    predicted_clusters = cluster_model.fit_predict(embeddings)

    num_clusters = len(set(predicted_clusters))
    print(f"Clusters found: {num_clusters}")

    correct = 0
    for cid in set(predicted_clusters):
        cluster_labels = [true_labels[i] for i, v in enumerate(predicted_clusters) if v == cid]
        dominant = max(set(cluster_labels), key=cluster_labels.count)
        correct += cluster_labels.count(dominant)

    print(f"Overall Accuracy (Purity): {(correct / len(embeddings)) * 100:.2f}%")
    print(f"ARI Score: {adjusted_rand_score(true_labels, predicted_clusters):.4f}")


# executie
if __name__ == "__main__":
    if HAS_DML:
        device = torch_directml.device()
        print(f"AMD GPU acceleration ENABLED via DirectML: {device}")
    else:
        device = torch.device("cpu")
        print("⚠torch-directml not installed. Defaulting to CPU.")

    model = TinyFaceNet().to(device)
    dataset_dir = "dataset"

    if os.path.exists(dataset_dir):
        # train
        train_model(model, device, dataset_dir, epochs=50, batch_size=128)

        # db
        client, collection_name = setup_qdrant()

        # 3.embedding si cluster
        embed_dataset(model, client, collection_name, dataset_path=dataset_dir)
        evaluate_clusters(client, collection_name)

        client.close()
    else:
        print(f"Error: Directory '{dataset_dir}' not found.")