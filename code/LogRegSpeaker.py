import pickle
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack
import numpy as np

DIR = "../datasets/shakespeare/"

PATH = DIR + "will_play_text.csv"
OUTPATH = DIR + "play_dict.p"
DATAPATH = DIR + "data.p"

d = pickle.load(open(OUTPATH, "rb"))[0]

LOAD = True
PCA = True

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
	if PCA:

	f = open(DATAPATH, "wb")
	pickle.dump([data, labels, mat, vectors], f)
	f.close()
else:
	f = open(DATAPATH, "rb")
	data, labels, mat, vectors = pickle.load(f)
	f.close()

labels = labels[:-1]
print("Distribution is: {0}".format(float(sum(labels))/len(labels)))

print("Splitting data")

TRAINING_NUM = 5000
p_s = [i for i in xrange(len(labels)) if labels[i] == 1]
p_data, p_labels = vectors[p_s, :], [i for i in p_s]
t_data, t_labels = vectors[:TRAINING_NUM], labels[:TRAINING_NUM]
v_data, v_labels = vectors[TRAINING_NUM+1:], labels[TRAINING_NUM+1:]

lr = LogisticRegression()
print("Begun Training")
lr.fit(t_data, t_labels)
print("The val accuracy is: " + str(lr.score(v_data, v_labels)))
print("The train accuracy is: " + str(lr.score(t_data, t_labels)))
print("The positive accuracy is: " + str(lr.score(p_data, p_labels)))
