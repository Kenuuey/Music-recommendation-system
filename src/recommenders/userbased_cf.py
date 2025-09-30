import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

class UserBasedCF:
    def __init__(self, k_neighbors=50, n_components=64, nlist=1000, nprobe=10):
        """
        k_neighbors : int
            Number of nearest neighbors to store for each user.
        n_components : int
            Dimension of user latent vectors after SVD.
        nlist : int
            Number of clusters in IVF (controls speed/accuracy tradeoff).
        nprobe : int
            Number of clusters to probe at query time.
        """
        self.k_neighbors = k_neighbors
        self.n_components = n_components
        self.nlist = nlist
        self.nprobe = nprobe

    def fit(self, user_item_sparse):
        # Step 1: Reduce dimensionality (dense latent user vectors)
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        user_latent = svd.fit_transform(user_item_sparse)

        # Step 2: Normalize for cosine similarity (dot product = cosine)
        faiss.normalize_L2(user_latent)

        # Step 3: Build FAISS IVF index
        d = user_latent.shape[1]
        quantizer = faiss.IndexFlatIP(d)  # inner product as similarity
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)

        
        ### 30.09 13:22

        # Train the index (clusters)
        index.train(user_latent.astype(np.float32))
        index.add(user_latent.astype(np.float32))

        # Set how many clusters to search
        index.nprobe = self.nprobe

        # Step 4: Perform search (ANN)
        D, I = index.search(user_latent.astype(np.float32), self.k_neighbors + 1)

        # Step 5: Store neighbors (skip self in results)
        self.user_sim = defaultdict(list)
        for u in range(len(I)):
            for neighbor, score in zip(I[u][1:], D[u][1:]):  # skip self
                if neighbor >= 0:
                    self.user_sim[u].append((neighbor, float(score)))

        print(f"✅ Built neighbor lists for {len(self.user_sim)} users "
              f"using IVF-Flat (nlist={self.nlist}, nprobe={self.nprobe}).")

        return self
