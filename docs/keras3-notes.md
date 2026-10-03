# Keras 3 notes for the recurrent-model notebooks

Colab runs Keras 3, which changes a few APIs that older tutorials use.

- `keras.preprocessing.text.Tokenizer` no longer exists. Use `keras.layers.TextVectorization(max_tokens=..., output_sequence_length=max_len)` and call `.adapt()` on the **training** texts only.
- `TextVectorization` lowercases and strips punctuation by default, so `<user>` becomes `user` and `#vaccine` becomes `vaccine`. Keep that only if notebook 01 justifies it; otherwise pass `standardize="lower"` or a custom function.
- Its vocabulary already reserves index 0 for padding and 1 for unknown words, so use `Embedding(input_dim=vectorizer.vocabulary_size(), ...)` with no extra +1.
- It keeps the first `max_len` tokens and pads at the end. `keras.utils.pad_sequences` pads and truncates at the start by default; with `mask_zero=True` the fast cuDNN GRU/LSTM kernel requires end padding (`padding="post"`).
- `recurrent_dropout` greater than 0 switches GRU and LSTM to a much slower non-cuDNN kernel; the `dropout` argument (on the layer inputs) keeps the fast kernel.
- Save models as `models/simple_rnn.keras`, `models/lstm.keras` and `models/gru.keras`. If `TextVectorization` sits outside the model, also save the settings that rebuild it (standardization, `max_tokens`, `max_len`). Don't commit the vocabulary itself: it is derived from the tweets, and adapting on the training split rebuilds it exactly. The one exception is `models/gru_glove_100d.npz` (notebook 04, round 4): it stores the GRU's 5,666-entry vocabulary with the 100-dimensional GloVe Twitter vector of each entry (zeros where GloVe has none) and how each was matched, so a rerun can check that the cached rows still line up with the rebuilt vocabulary and skip the 1.5 GB GloVe download.
- scikit-learn 1.6 on Colab has no `mean_squared_error(squared=False)`. Use `common.rmse`.
