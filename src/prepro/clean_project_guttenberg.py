import os
import numpy as np
import multiprocessing as mp
import argparse

def clean(args):
	(filename, outfilename) = args

	with open(filename, encoding = "ISO-8859-1") as f:
		text = f.read()

	start_idx = text.find("***")
	
	if start_idx == -1:
		print (filename)
		return

	text = text[start_idx:]
	start_idx = text.find("\n")
	text = text[start_idx:]

	end_idx = text.find('*** END OF THIS PROJECT GUTENBERG EBOOK')
	text = text[:end_idx]

	paragraphs = text.split('\n\n')
	paragraphs = [p.replace('\n',' ') for p in paragraphs]

	text = "\n".join(paragraphs)

	with open(outfilename,'w') as f:
		f.write(text)

def clean_files(args):
	files = os.listdir(args.idir)
	files = [fn for fn in files 
				if fn.split('.')[-1] == args.iext]
	print(len(files))
	iFiles = [os.path.join(args.idir, fn) for fn in files]
	oFiles = [os.path.join(args.odir, ".".join(fn.split('.')[:-1]+[args.oext])) for fn in files]

	if not os.path.exists(args.odir):
		os.makedirs(args.odir)	

	p = mp.Pool(mp.cpu_count())
	p.map(clean, zip(iFiles, oFiles))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--idir')
	parser.add_argument('--odir')
	parser.add_argument('--iext')
	parser.add_argument('--oext', type=str, default='')

	args = parser.parse_args()
	print(clean_files(args))

