import transformers

MAX_LEN = 200
DEVICE = "cuda:0"
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
MODEL_PATH = "/inputs/model.bin"
TRAINING_FILE = "/inputs/train.csv"