### How to use binomial_checkpoint

binomial_checkpoint acccepts a function, a postprocessing function, a number of loop steps, and a number of checkpoints, and returns an autograd primitive which can be applied to parameters, an initial state, and a sequence of inputs to yield the  postprocessed states resulting from looped computation of the parametrized function over the inputs and starting at the initial state. 

It takes functions of the following signature

`` f(parameters, state, input) -> state ``

Some examples of different configurations of f are as follows. 

An LSTM requires a hidden and cell state along with input to propagate to the next state. The output of the LSTM over a sequence of input is given by a sequence of output probabilities, but since the output probabilities are independent of the cell state given the hidden state, we can represent the transformation from hidden state to output probabilities as the postprocessing function and the function f as the LSTM which propagates from `(cell_state, hidden_state)` to `(cell_state, hidden_state)` given an `input`. 

The other arguments to binomial_checkpoint are self explanatory — number of checkpoints to save during reverse-mode AD and number of steps in the loop. 

Calling grad on binomial_checkpoint returns a harcoded derivative which performs the checkpointing algorithm of Griewank (2000)

### RNN Script

A demo of binomial_checkpoint applied to a simple RNN is available in this directory as an interactive notebook. Performance benchmarks are computed inline. 

### LSTM Scripts

We can analyze the memory requirements of an LSTM with and without checkpointing by comparing the overhead introduced in each case relative to an identical script with minimized memory requirements. 

First, we can look at the output of top for a dummy baseline script that just returns zero for the gradient of the loss function. We see Python is using about 121 mb of memory. 
```
PID   COMMAND      %CPU  TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    
8591  Python       134.1 01:29.49 2/1   0    15    121M+  0B     0B     8591 7415 running
```

Let's use normal_lstm.py to compute the gradient normally
```
PID   COMMAND      %CPU  TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    
6136  Python       197.1 00:20.51 2/2   0    15    362M+  0B     0B     6136 5987 running
```

As we would expected, the memory demands are a lot higher. The checkpointed script does better. 
```
PID   COMMAND      %CPU  TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    
6140  Python       141.1 00:48.66 2/1   0    15    151M+  0B     0B     6140 5987 running
```

Per iteration, the checkpointed LSTM takes about 3.30 seconds, while the uncheckpointed LSTM takes 
about 2.05 seconds. The checkpointed LSTM introduces 85 mb of memory over baseline, while the normal
LSTM introduces 284 mb of memory. This corresponds to a slowdown of 1.62x, with memory savings of 9.47x.


