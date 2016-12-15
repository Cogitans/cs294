import csv
import numpy as np
import random

SHAKESPEARE_VALIDATION_PLAYS = ['Taming of the Shrew', 'The Tempest',
								'Timon of Athens', 'Titus Andronicus', 
								'Troilus and Cressida', 'Twelfth Night', 
								'Two Gentlemen of Verona', 'A Winters Tale']

def shakespeare_raw_train_gen():
	"""
	Generator for raw train data from Shakespeare data set
	Yields pairs of (line, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if the current speaker is also 
			the speaker of the previous line yield. 1 if it is not
	"""
	with open('../datasets/shakespeare/will_play_text.csv', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			linenum, play, charnum, line, char, line = row
			if play in SHAKESPEARE_VALIDATION_PLAYS:
				continue
			if len(line) == 0 or charnum == '':
				continue
			did_speaker_change = 1 if prev_speaker != charnum and prev_speaker else 0
			prev_speaker = charnum
			yield line, did_speaker_change

def shakespeare_raw_test_gen():
	"""
	Generator for raw test data from Shakespeare data set
	Yields pairs of (line, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if the current speaker is also 
			the speaker of the previous line yield. 1 if it is not
	"""
	with open('../datasets/shakespeare/will_play_text.csv', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			linenum, play, charnum, line, char, line = row
			if play not in SHAKESPEARE_VALIDATION_PLAYS:
				continue
			if len(line) == 0 or charnum == '':
				continue
			did_speaker_change = 1 if prev_speaker != charnum and prev_speaker else 0
			prev_speaker = charnum
			yield line, did_speaker_change


def shakespeare_soft_train_gen():
	"""
	Generator for train data from Shakespeare data set
	Processes the output from raw_train_gen and yields 
	triples of (line1, line2, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if line1 and line2 are spoken
			by the same speaker, 1 is not
	"""
	raw_gen = shakespeare_raw_train_gen()
	prev_line = next(raw_gen)[0]
	samples = []
	for line in raw_gen:
		samples.append((prev_line, line[0], line[1]))
		prev_line = line[0]
	random.shuffle(samples)
	for sample in samples:	
		yield sample

def shakespeare_soft_test_gen():
	"""
	Generator for test data from Shakespeare data set
	Processes the output from raw_train_gen and yields 
	triples of (line1, line2, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if line1 and line2 are spoken
			by the same speaker, 1 is not
	"""
	raw_gen = shakespeare_raw_test_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		yield (prev_line, line[0], line[1])
		prev_line = line[0]

def wilde_raw_gen():
	"""
	Generator for train data from Wilde data set
	Yields pairs of (line, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if the current speaker is also 
			the speaker of the previous line yield. 1 if it is not
	"""
	for i in xrange(1,4):
		with open('../datasets/wilde_{}_parsed.txt'.format(i), 'r') as f:
			parsed_csv = csv.reader(f, delimiter=';')
			prev_speaker = None
			for row in parsed_csv:
				if len(row) < 2:
					continue
				if len(row) > 2:
					speaker = row[0]
					line = '. '.join(row[1:])
				else:
					speaker, line = row
				did_speaker_change = 1 if prev_speaker != speaker and prev_speaker else 0
				prev_speaker = speaker
				yield line, did_speaker_change

def wilde_soft_gen():
	"""
	Generator for data from Wilde data set
	Processes the output from raw_train_gen and yields 
	triples of (line1, line2, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if line1 and line2 are spoken
			by the same speaker, 1 is not
	"""
	raw_gen = wilde_raw_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		yield (prev_line, line[0], line[1])
		prev_line = line[0]

def movie_raw_gen():
	"""
	Generator for raw data from movie data set
	Yields pairs of (line, did_speaker_change)
		- line is the next line in the play
		- did_speaker_change is 0 if the current speaker is also 
			the speaker of the previous line yield. 1 if it is not
	"""
	with open('../datasets/movies.txt', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			if len(row) == 1 and row[0] == 'BREAK':
				prev_speaker = None
				continue
			if len(row) > 2:
				speaker = row[0]
				line = '. '.join(row[1:])
			else:
				speaker, line = row
			if len(line) == 0:
				continue
			did_speaker_change = 1 if prev_speaker != speaker and prev_speaker else 0
			prev_speaker = speaker
			yield line, did_speaker_change

def movie_soft_train_gen():
	"""
	Generator for train data from movie data set
	Processes the output from raw_train_gen and yields 
	triples of (line1, line2, did_speaker_change)
		- line is the next line in the movie
		- did_speaker_change is 0 if line1 and line2 are spoken
			by the same speaker, 1 is not
	"""
	samples = []
	raw_gen = movie_raw_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		samples.append((prev_line, line[0], line[1]))
		prev_line = line[0]
	for i in range(int(0.8*len(samples))):
		yield samples[i]

def movie_soft_test_gen():
	"""
	Generator for test data from movie data set
	Processes the output from raw_train_gen and yields 
	triples of (line1, line2, did_speaker_change)
		- line is the next line in the movie
		- did_speaker_change is 0 if line1 and line2 are spoken
			by the same speaker, 1 is not
	"""
	samples = []
	raw_gen = movie_raw_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		samples.append((prev_line, line[0], line[1]))
		prev_line = line[0]
	for i in range(int(0.8*len(samples)), len(samples)):
		yield samples[i]
