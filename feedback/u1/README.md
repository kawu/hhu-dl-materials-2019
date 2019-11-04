# Homework

https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session1/u1_eng.pdf


# Common mistakes

(Some of these are not really mistakes, but do not follow Python conventions or
are otherwise problematic)

### Train/dev/test not disjoint

Given dataset list `data`, things like this:
```python
part1 = random.sample(data, size_of_part1)
part2 = random.sample(data, size_of_part2)
```
Since the two samples are taken independently, the same element can end up in
both `part1` and `part2`.

### Using random.choices

The exercise can be solved using `random.shuffle` (probably the easiest way) or
`random.sample`, but `random.choices` gives a sample *with replacement* which
may mess up the numbers of occurrences of the different elements in the list
that we want to split.

### Mishandling repetitions

Here's a code fragment that performs a random two-way split given a `data`
list.  If an element `x` occurs several times in `data` and exactly one of its
occurrences ends up in `part1`, all the other occurrences of `x` will be
discarded.
```python
part1 = random.sample(data, int(rel_size * len(data)))
part2 = [i for i in data if i not in part1]
```

It's possible to use `random.sample` to solve the exercise by applying it to
the list of indices.
```python
ixs = range(len(data))
part1_ixs = random.sample(ixs, int(rel_size * len(data)))
part2_ixs = [i for i in ixs if i not in part1_ixs]
```

### Naming conventions

This piece of code is problematic (even if nominally correct) because
`DataDict` is first defined as a type, then overwritten and assigned a value
(an actual dictionary with names).
```python
DataDict = Dict[Lang, List[Name]]
...
DataDict = read_names(names_path)
```
It's better to stick to the Pythonic naming convention of using:
* CamelCase for classes and types
* under\_score for values, functions, methods, etc.
(did I miss something?)

### Shuffling in place

A pretty common issue was to shuffle the data list in place.  This means that
`random_split` has a non-documented side effect.  It didn't matter in this
exercise, but in a larger application this could lead to some hard-to-identify
bugs.
```python
def random_split(data: list, abs_size: int) -> Tuple[list, list]:
    ...
    random.shuffle(data)
```

### Misusing types

Be careful about type annotations, Python does not check them by default.  The
following is confusing, because `{}` means that e.g. `data_train` is a
dictionary, not a tuple.
```python
data_train: tuple = {}
data_dev: tuple = {}
data_test: tuple = {}
```

# Misc

### Balanced split for each language

In (at least) one solution, the dataset was split to train/dev/test for each
language separately.  While this is not exactly consistent with the description
of the exercise (the distribution of splits is not uniform), it does the job.
Is it actually better?

