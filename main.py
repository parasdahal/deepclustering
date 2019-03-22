from data import load_data
from model import ClusterNetwork

### MNIST
# -------------------------------- 
# Clustering Accuracy: 0.97019
# NMI: 0.92337
# Adjusted Rand Index: 0.93545
# -------------------------------- 

X, y = load_data('mnist')

mnist = ClusterNetwork(
    latent_weight = 0.01,
    optimizer='sgd_mom',
    learning_rate=0.001,
    latent_dim=10,
    alpha1= 20
).train(X, y, train_batch_size=100, pretrain_epochs=10, train_epochs=50)

### Fashion MNIST
# -------------------------------- 
# Clustering Accuracy: 0.6649
# NMI: 0.67089
# Adjusted Rand Index: 0.52909
# -------------------------------- 

X, y = load_data('fmnist')

fmnist = ClusterNetwork(
    latent_dim = 10,
    latent_weight = 0.01,
    noise_factor = 0.5,
    keep_prob = 1.0,
    alpha1=20,
    alpha2=1,
    optimizer='adam',
    learning_rate=0.001
).train(X, y, train_batch_size=100, pretrain_epochs=10, train_epochs=200)


### Reuters
# -------------------------------- 
# Clustering Accuracy: 0.83616
# NMI: 0.63712
# Adjusted Rand Index: 0.66342
# -------------------------------- 

X, y = load_data('reuters')

reuters = ClusterNetwork(
    latent_dim = 10,
    latent_weight = 0.001,
    noise_factor = 0.4,
    keep_prob = 1.0,
    alpha1=20,
    alpha2=1,
    optimizer='adam',
    learning_rate=0.001,
    n_clusters=4,
).train(X, y, train_batch_size=100, pretrain_epochs=5, train_epochs=50)

### STL
# -------------------------------- 
# Clustering Accuracy: 0.88723
# NMI: 0.78844
# Adjusted Rand Index: 0.76962
# -------------------------------- 

X, y = load_data('stl')

stl = ClusterNetwork(
    latent_dim = 10,
    latent_weight = 0.001,
    noise_factor = 0.4,
    keep_prob = 1.0,
    alpha1=20,
    alpha2=1,
    optimizer='adam',
    learning_rate=0.001,
    n_clusters=10,
).train(X, y, train_batch_size=100, pretrain_epochs=20, train_epochs=100)



### USPS
# -------------------------------- 
# Clustering Accuracy: 0.88745
# NMI: 0.80439
# Adjusted Rand Index: 0.80219
# -------------------------------- 
X, y = load_data('usps')
print(X.shape, y.shape)

usps = ClusterNetwork(
    latent_dim = 10,
    latent_weight = 0.001,
    noise_factor = 0.2,
    alpha1=20,
    alpha2=1,
    optimizer='adam',
    learning_rate=0.001,
).train(X, y, train_batch_size=100, pretrain_epochs=50, train_epochs=0)




# Plotting

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10,16))
    ax = plt.subplot(111)
    ax.set_facecolor("white")
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

# z_state = np.load('z_state.npy')
# print(z_state.shape)


# kmeans = KMeans(n_clusters=10, n_init=20)
# y_pred = kmeans.fit_predict(z_state)
# acc = cluster_acc(y_test.argmax(1), y_pred)
# print('Clustering ACC: '+str(acc))

# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2).fit_transform(z_state[:5000])

# plot_embedding(X_embedded[:5000] ,y_test[:5000].argmax(1))

z_state = np.load('zl_state.npy')
print(z_state.shape)


kmeans = KMeans(n_clusters=10, n_init=20)
y_pred = kmeans.fit_predict(z_state)
acc = cluster_acc(y_test.argmax(1), y_pred)
print('Clustering ACC: '+str(acc))

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(z_state[:5000])

plot_embedding(X_embedded[:5000],y_test[:5000].argmax(1))