# Keras ByteNet Decoder for text generation
Implementing text generation using only convolution layers with dilation and residual layers
(ByteNet decoder [Link to paper](https://arxiv.org/pdf/1610.10099.pdf)) in order to predict the next character with
linear time
Similar to lstm_text_generation taken from keras examples

2 options: w/o embedding layer, although the input is characters based, the embedding of numbers and punctuation should
 be quite similar accordingly


## Dataset used
nietzsche.txt from

