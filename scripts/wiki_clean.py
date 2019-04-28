import re

# data_directory = "/home/hyd/workhorse2/narrativeQA/NN4NLP-Project/wikitext-103/"
# train_data = data_directory + "wiki.train.tokens"
# valid_data = data_directory + "wiki.valid.tokens"
# test_data = data_directory + "wiki.test.tokens"

# train_new_data = data_directory + "wiki.train.tokens.new"
# valid_new_data = data_directory + "wiki.valid.tokens.new"
# test_new_data = data_directory + "wiki.test.tokens.new"

# train_map_data = data_directory + "wiki.train.tokens.mapped"
# valid_map_data = data_directory + "wiki.valid.tokens.mapped"
# test_map_data = data_directory + "wiki.test.tokens.mapped"

# train_data_vocab = data_directory + "wiki.train.vocab"
# valid_data_vocab = data_directory + "wiki.valid.vocab"
# test_data_vocab = data_directory + "wiki.test.vocab"



## penn tree bank data 
data_directory = "/home/hyd/workhorse2/narrativeQA/NN4NLP-Project/LM_data/ptb/"
train_new_data = data_directory + "ptb.train.txt"
valid_new_data = data_directory + "ptb.valid.txt"
test_new_data = data_directory + "ptb.test.txt"

train_map_data = data_directory + "ptb.train.mapped"
valid_map_data = data_directory + "ptb.valid.mapped"
test_map_data = data_directory + "ptb.test.mapped"

train_data_vocab = data_directory + "ptb.train.vocab"
valid_data_vocab = data_directory + "ptb.valid.vocab"
test_data_vocab = data_directory + "ptb.test.vocab"


# regex_to_use = "[ =]+ "
regex_to_use = "=([^=]*)="

def clean_wiki_data():
	with open(test_data, 'r') as f:
		text = f.read()
		text = re.sub(regex_to_use, "", text)
		text = text.replace('\n',  ' ').replace('\r', '')
		with open(test_new_data, 'w') as fnew:
			fnew.write(text)

	with open(valid_data, 'r') as f:
		text = f.read()
		text = re.sub(regex_to_use, "", text)
		text = text.replace('\n',  ' ').replace('\r', '')
		with open(valid_new_data, 'w') as fnew:
			fnew.write(text)
	print("done with valid")
	with open(train_data, 'r') as f:
		text = f.read()
		text = re.sub(regex_to_use, "", text)
		text = text.replace('\n',  ' ').replace('\r', '')
		with open(train_new_data, 'w') as fnew:
			fnew.write(text)

def get_vocab():
	train_vocab=set([])
	with open(train_new_data, 'r') as f:
		text = f.read()
		text = text.replace('\n',  ' ').replace('\r', '')
		train_text = text.split()
		train_vocab.update(train_text)
	print("train tokens: %d" %len(train_text))
	print("train vocab: %d" %(len(train_vocab)))
	mapping = dict(zip(train_vocab, range(len(train_vocab))))
	train_mapped = [str(mapping[word]) for word in train_text]
	with open(train_data_vocab, 'w') as fnew:
		for key, value in mapping.items():
			fnew.write(key +" " + str(value) + "\n")

	with open(train_map_data, 'w') as fnew:
		text = ' '.join(train_mapped)
		fnew.write((text))

	#########################################

	valid_vocab=set([])
	with open(valid_new_data, 'r') as f:
		text = f.read()
		text = text.replace('\n',  ' ').replace('\r', '')
		valid_text = text.split()
		valid_vocab.update(valid_text)
	print("valid tokens: %d" %len(valid_text))
	print("valid vocab: %d" %(len(valid_vocab)))
	mapping = dict(zip(valid_vocab, range(len(valid_vocab))))
	valid_mapped = [str(mapping[word]) for word in valid_text]
	with open(valid_data_vocab, 'w') as fnew:
		for key, value in mapping.items():
			fnew.write(key +" " + str(value) + "\n")

	with open(valid_map_data, 'w') as fnew:
		text = ' '.join(valid_mapped)
		fnew.write((text))

	#########################################

	test_vocab=set([])
	with open(test_new_data, 'r') as f:
		text = f.read()
		text = text.replace('\n',  ' ').replace('\r', '')
		test_text = text.split()
		test_vocab.update(test_text)
	print("test tokens: %d" %len(test_text))
	print("test vocab: %d" %(len(test_vocab)))
	mapping = dict(zip(test_vocab, range(len(test_vocab))))
	test_mapped = [str(mapping[word]) for word in test_text]
	
	with open(test_data_vocab, 'w') as fnew:
		for key, value in mapping.items():
			fnew.write(key +" " + str(value) + "\n")

	with open(test_map_data, 'w') as fnew:
		text = ' '.join(test_mapped)
		fnew.write((text))

	return 

# clean_wiki_data()
get_vocab()