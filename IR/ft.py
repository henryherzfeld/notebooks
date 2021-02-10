import fasttext as ft
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


model = ft.train_unsupervised('moby_dick.txt', model='skipgram')
pca = PCA(n_components=2)

reduced = pca.fit_transform(model.words)


plt.()
