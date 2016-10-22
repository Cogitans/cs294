import pickle
from sklearn.decomposition import PCA, SparsePCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import vstack
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, LabeledSentence
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential, Model, load_model
from keras.layers import Merge, Input, merge
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Activation, Reshape, Flatten
from scipy.sparse import vstack

DIR = "../datasets/shakespeare/"

PATH = DIR + "will_play_text.csv"
OUTPATH = DIR + "play_dict.p"
DATAPATH = DIR + "data.p"

d = pickle.load(open(OUTPATH, "rb"))[0]

LOAD = False
do_PCA = False
do_DOC = False
RECUR = False

if LOAD:
	data = []
	labels = []
	sentences = []

	for play in d:
		for i in xrange(len(d[play][0])-1):
			data.append([d[play][0][i], d[play][0][i+1]])
			sentences.append(d[play][0][i])
			labels.append(0 if d[play][1][i] == d[play][1][i+1] else 1)


	vect = CountVectorizer()
	mat = vect.fit_transform(sentences)

	vectors = vstack(tuple([mat[i, :] - mat[i+1, :] for i in xrange(mat.shape[0]-1)]))
	model = None
	if do_DOC:
		print("Doc-ing")
		model = Doc2Vec([LabeledSentence(sentences[i].split(" "), [str(i)]) for i in range(len(sentences))])
		mat = np.empty((1, 300))
		print("Here")
		i = 0
		tostack = []
		for sent in sentences:
			if i % 1000 == 0:
				print(i)
			tostack.append(np.reshape(model.infer_vector(sent), (1, 300)))
			i += 1
		mat = np.vstack(tostack)
		vectors = vstack(tuple([mat[i, :] - mat[i+1, :] for i in xrange(1, mat.shape[0]-1)]))
		print("THERE")


	f = open(DATAPATH, "wb")
	pickle.dump([data, labels, mat, vectors, sentences, model], f)
	f.close()
else:
	f = open(DATAPATH, "rb")
	data, labels, mat, vectors, sentences, model = pickle.load(f)
	f.close()

if do_PCA:
	pca = SparsePCA(2)
	M = 100
	v = vectors[0:M, :]
	print(v.shape)
	projected_matrix = pca.fit_transform(v.toarray())
	print("Fit")
	l = labels[:-1]
	l = l[:M]
	plt.clf()
	p_s = [i for i in xrange(len(l)) if l[i] == 1]
	f_s = [i for i in xrange(len(l)) if l[i] != 1]
	differens = projected_matrix[p_s, :]
	sames = projected_matrix[f_s, :]
	plt.plot(sames[:, 0], sames[:, 1], "b*")
	plt.plot(differens[:, 0], differens[:, 1], "r*")
	plt.show()

labels = labels[:-1]
print("Distribution is: {0}".format(float(sum(labels))/len(labels)))

print("Splitting data")

TRAINING_NUM = 5000
p_s = [i for i in xrange(len(labels)) if labels[i] == 1]
n_s = [i for i in xrange(len(labels)) if labels[i] == 0]
p_data, p_labels = vectors[p_s, :], [1 for i in p_s]
n_data, n_labels = vectors[n_s, :], [0 for i in n_s]
even_p_data, even_p_labels = p_data[0:30000, :], p_labels[0:30000]
even_n_data, even_n_labels = n_data[0:30000, :], n_labels[0:30000]
print(even_n_data.shape)
print(even_p_data.shape)

pn_data = vstack([even_n_data, even_p_data])
print(pn_data.shape)
pn_labels = even_n_labels + even_p_labels

mask = np.random.permutation(len(pn_labels))

pn_data = pn_data[mask, :]
pn_labels = np.array(pn_labels)[mask, :]

# pn_data = np.empty((even_n_data.shape[0] + even_p_data.shape[0], even_p_data.shape[1]))
# pn_data[::2, :] = even_p_data
# pn_data[1::2, :] = even_n_data
# pn_labels = [even_n_data[i] if i % 2 == 0 else even_p_data[i] for i in range(pn_data.shape[0])]
t_data, t_labels = vectors[:TRAINING_NUM], labels[:TRAINING_NUM]
v_data, v_labels = vectors[TRAINING_NUM+1:], labels[TRAINING_NUM+1:]




lr = RandomForestClassifier(n_estimators=100)
print("Begun Training")
lr.fit(pn_data, pn_labels)
print("The val accuracy is: " + str(lr.score(v_data, v_labels)))
print("The train accuracy is: " + str(lr.score(t_data, t_labels)))
print("The positive accuracy is: " + str(lr.score(p_data, p_labels)))
print("The negative accuracy is: " + str(lr.score(n_data, n_labels)))

if RECUR:
	MID_LAYER = 50
	network = Sequential()
	network.add(GRU(MID_LAYER, input_dim=(mat.shape)))
	network.add(GRU(1))
	network.compile(optimizer="adam")
	network.fit(mat, labels)