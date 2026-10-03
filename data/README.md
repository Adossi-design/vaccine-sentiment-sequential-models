# data/

Put `Train.csv` here. It is never committed: Zindi's rules forbid publishing it. See [docs/data.md](../docs/data.md) for how to get it.

Notebook 04's round 4 can also use `glove.twitter.27B.100d.txt` (GloVe Twitter vectors, Pennington et al., 2014) from here. You only need it if `models/gru_glove_100d.npz` is missing: that file, saved next to the models and committed with them, holds the GloVe rows for the GRU's vocabulary, and the notebook downloads the 1.5 GB archive into this folder only when it has to rebuild it. Like `Train.csv`, the GloVe file is ignored by git.
