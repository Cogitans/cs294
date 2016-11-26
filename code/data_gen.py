import csv

def shakespeare_raw_gen():
	with open('../datasets/shakespeare/will_play_text.csv', 'r') as f:
		parsed_csv = csv.reader(f, delimiter=';')
		prev_speaker = None
		for row in parsed_csv:
			linenum, play, charnum, line, char, line = row
			if len(line) == 0 or charnum == '':
				continue
			did_speaker_change = 1 if prev_speaker != charnum and prev_speaker else 0
			prev_speaker = charnum
			yield line, did_speaker_change


def shakespeare_soft_get():
	raw_gen = shakespeare_raw_gen()
	prev_line = next(raw_gen)[0]
	for line in raw_gen:
		yield (prev_line, line[0], line[1])
		prev_line = line[0]

