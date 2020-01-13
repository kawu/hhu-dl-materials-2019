# Dependencies

This document describes the modifications of the code introduced in order to
handle dependencies.  This current code is still a scaffolding, the actual
model for the prediction of dependency heads is not implemented yet.


## Data

The first modification involves the representation of the dataset.  Namely,
Token is now defined as a [record with information about the word, the POS tag,
and the corresponding dependency
head](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/data.py#L8-L26).
The record attributes can be simply accessed using the dot, as exemplified [in
the POS tagging loss
function](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L332-L333).

We could extend the Token record with information about the dependency head
label (subject, object, etc.).  This should require no additional refactoring
(i.e., no modifications in other places of the code), which is a good reason to
use a named tuple instead of a regular tuple.


## Tagger

The implementation of the tagger has been moved from the `main.py` module to
the separate `tagger.py` module.

#### Scores

The implementation of the scoring model, as before, is divided into two parts.
In the [first
part](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L49-L105),
scoring for individual sentences is implemented.  In the [second
part](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L107-L187)
sentences are processed in batches.  The latter is used by default, but the
simpler version (which works over single sentences) allows to check if the
optimized version works as expected.

The [forward
function](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L89-L105)
now returns a pair of score tensors: POS tagging-related and dependency
parsing-related scores, respectively.  Both types of scores are calculated
based on the same, [contextualized embedding
vectors](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L53-L67).
All the dependency-related scores are currently [set to
0](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L80-L87).

#### Tagging

The tagger [predicts the POS tags and dependency heads
jointly](TODO).
The [head prediction
subroutine](TODO)
is already implemented and works similarly to the [POS tag prediction
subroutine](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L219-L239).

#### Accuracy

POS tagging accuracy is calculated with the [pos_accuracy
function](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L260-L288),
while the head prediction accuracy is calculated with the [dep_accuracy
function](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L291-L319).
Again, the latter is virtually a copy of the former (some common code fragments
could be probably factored out).

#### Loss

The previously called `total_loss` function, which calculates the cross-entropy
loss of the model on the given dataset, is now renamed as
[pos_loss](https://github.com/kawu/hhu-dl-materials/blob/897a6ad472389b406f857c2eef12573d79ba383c/universal-pos-deps/tagger.py#L322-L356).
Since our goal is to predict both POS tags and dependency heads, we should
extend this function to capture both POS-related and dependency-related loss.
