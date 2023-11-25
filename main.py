from utils import Envirodataset

train=Envirodataset('data/NextDayPred/archive/train.jsonl')
dev=Envirodataset('data/NextDayPred/archive/dev.jsonl')
test=Envirodataset('data/NextDayPred/archive/test.jsonl')

print(len(train), len(dev), len(test))
