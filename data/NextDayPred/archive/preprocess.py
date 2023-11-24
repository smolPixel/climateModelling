from torchdata.datapipes.iter import FileLister, FileOpener
import json

set='test'

datapipe1 = FileLister(".", "*.tfrecord")
datapipe2 = FileOpener(datapipe1, mode="b")
tfrecord_loader_dp = datapipe2.load_from_tfrecord()
print('hello')


out=open(f'{set}.jsonl', 'a+')

i=0
for example in tfrecord_loader_dp:
	i+=1
	print(i)
	example={key: item.tolist() for key, item in example.items()}
	jj=json.dumps(example)
	out.write(jj)
	out.write('\n')
	# print(example)
	# fds
print(i)
print('done')
print('done')