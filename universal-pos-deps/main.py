
from neural.training import train
import data
from tagger import Tagger, dep_accuracy, total_loss
from word_embedding import FastText


# Training dataset
train_set = data.MemPosDataSet(
    "UD_English-ParTUT/en_partut-ud-train.conllu",
    sort_by_len=True
)

# Development dataset
dev_set = data.MemPosDataSet("UD_English-ParTUT/en_partut-ud-dev.conllu")

# Size stats
print("Train size:", len(list(train_set)))
print("Dev size:", len(list(dev_set)))

# Determine the set of words in the dataset
word_set = set(
    tok.word
    for sent in train_set
    for tok in sent
)

# Number of words
print("Number of words:", len(word_set))

# Determine the POS tagset
tagset = set(
    tok.upos
    for sent in train_set
    for tok in sent
)

# Tagset
print("Tagset:", tagset)

# Create the word embedding module
# word_emb = AtomicEmbedder(word_set, 10)
word_emb = FastText(
    "wiki-news-300d-1M-subword-selected.vec",
    limit=10**5,    # The maximum number of words to load
    dropout=0.25
)

# Create the tagger
tagger = Tagger(word_emb, tagset, hid_size=200, hid_dropout=0.5)

# Train the model (see `train` in `neural/training`)
train(
    tagger, train_set, dev_set,
    total_loss, dep_accuracy,
    epoch_num=60,
    learning_rate=0.01,
    report_rate=10
)

# Second training phase (with lower learning-rate)
train(
    tagger, train_set, dev_set,
    total_loss, dep_accuracy,
    epoch_num=20,
    learning_rate=0.001,
    report_rate=10
)
