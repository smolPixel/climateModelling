from torchdata.datapipes.iter import FileLister, FileOpener
import json

set='train'

datapipe1 = FileLister(".", "*.tfrecord")
datapipe2 = FileOpener(datapipe1, mode="b")
tfrecord_loader_dp = datapipe2.load_from_tfrecord()
print('hello')

i=0
for example in tfrecord_loader_dp:
	i+=1
	jj=json.dumps(example)
	print(jj)
	fds
	# print(example)
	# fds
print(i)
print('done')
print('done')