# Homework

https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session8/u8_eng.pdf

The corresponding code with the solutions to the exercises can be found at: https://github.com/kawu/hhu-dl-materials/tree/master/universal-pos-deps

# Exercise 1

(Partially) solved in class.

See the implementation of the `Embedding` class in `neural/embedding.py`.


# Exercise 2

Have a look at the:
* Implementation of `AtomicEmbedder` in the
  `universal-pos-deps/word_embedding.py` file.
* Correction of the `forward` method of the `Embedding` class in
  `neural/embedding.py`, which allows to handle out-of-vocabulary words.

Another solution would be to copy the code from the `Embedding` class in
`neural/embedding.py` and reuse it to implement the `AtomicEmbedder`.  This,
however, would nake the implementation of the `AtomicEmbedder` less transparent
(since it would then mixe the generic with the specific functionality).

TODO: add links.


# Exercise 3

The issue with the '\_' tag is related to [tokenization and word
segmentation](https://universaldependencies.org/u/overview/tokenization.html)
in UD.  This tag is assigned to tokens that are not part of the selected tokenization.
See also https://universaldependencies.org/format.html#words-tokens-and-empty-nodes.

In our English dataset, '\_' is mostly assigned to contractions such as
"don't", "cannot", "aren't", etc.  It makes sense to simply discard them during
data loading.  The tokenizer should have no trouble identifying and splitting
them into component words.  Note also that the component words (e.g., "cannot"
-> "can" and "not") are already present in our UD files, so we don't have to
split such contractions ourselves.

Note, however, that this solution may not be optimal for some other (UD)
languages, where tokenization is not so obvious.  In particular, there's one
occurrence of "des" in our training dataset, which is a part of a French proper
name ("Vicaire des Ardennes").  "des" in French is ambiguous ("des" vs "des" vs
"de+les").
