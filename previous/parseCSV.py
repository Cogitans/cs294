DIR = "../datasets/shakespeare/"

PATH = DIR + "will_play_text.csv"
OUTPATH = DIR + "play_dict.p"

import csv
import pickle

texts = {}

with open(PATH, "rb") as csvfile:
	read = csv.reader(csvfile, delimiter=';', )
	for row in read:
		linenum, play, charnum, line, char, line = row
		if len(line) == 0:
			continue
		if play not in texts:
			texts[play] = ([], [])
		texts[play][0].append(line)
		texts[play][1].append(charnum)

f = open(OUTPATH, "wb")
pickle.dump([texts], f)
f.close()


		