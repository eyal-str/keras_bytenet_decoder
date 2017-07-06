# Keras ByteNet Decoder for text generation
Implementing text generation using only convolution layers with dilation and residual layers
(ByteNet decoder [Link to paper](https://arxiv.org/pdf/1610.10099.pdf)) in order to predict the next character with
linear time
Similar to lstm_text_generation taken from keras examples

## 4 Options:
1) ByteNet decoder w/o embedding layer - residual_dilated_conv_text_generation
2) ByteNet decoder with embedding layer - residual_dilated_conv_text_generation_embedding
3) ByteNet decoder predicting n next characters instead of just 1 - residual_dilated_conv_n_chars_text_generation
4) ByteNet decoder for the last n characters of the string concatenated with embedding of the whole sentence -
wide_view_residual_dilated_conv_text_generation

## Dataset used
nietzsche.txt from

