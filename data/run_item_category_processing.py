import pickle
import os
import numpy as np

# This fallback script expects an existing item embedding file and performs KMeans to generate item categories
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None
dataset = 'yelp'
def main():
    base = os.path.join(os.path.dirname(__file__), dataset, 'handled')
    itm_path = os.path.join(base, 'itm_emb_np.pkl') 
    out_path = os.path.join(base, 'item_label50.pkl')
    if not os.path.exists(itm_path):
        raise FileNotFoundError(itm_path)
    emb = pickle.load(open(itm_path, 'rb'))
    if isinstance(emb, np.ndarray):
        X = emb
    else:
        X = np.array(emb)
    K = 50
    if KMeans is None:
        # simple PCA k-means replacement using random centers
        np.random.seed(42)
        centers = X[np.random.choice(X.shape[0], K, replace=False)]
        # assign by nearest center
        dists = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        labels = dists.argmin(axis=1)
    else:
        km = KMeans(n_clusters=K, random_state=42)
        labels = km.fit_predict(X)
    data = {
        'labels': labels,
        'num_categories': int(labels.max()) + 1 if labels.size > 0 else 1
    }
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print('Saved item_label.pkl to', out_path)

if __name__ == '__main__':
    main()

