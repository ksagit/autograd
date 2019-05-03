### How to use binomial_checkpoint

binomial_checkpoint acccepts a function, a number of loop steps, and a number of checkpoints, and returns an autograd primitive which can be applied to parameters, an initial state, and a sequence of inputs to yield the result of looped computation of the parametrized function over the input starting at the initial state. Specifically, it takes functions of the following signature

`` f(parameters, (hidden_state, visible_state), input) -> (hidden_state, visible_state) ``

The state (second argument) is a tuple comprised of two things: the "visible state", which is everything we want to see in the output, and the hidden state, which is the stuff we don't need in the output, but which is necessary to propagate to the next state. For optimal performance, it is important to carefully choose the hidden and visible state. 

Some examples of different configurations of f are as follows. 

An LSTM requires a hidden and cell state along with input to propagate to the next state. Confusingly, the hidden state in the LSTM is the "visible state" in this setup, because the hidden state is what is used to compute the output probabilities — so we need it in the output. Since the output probabilities are independent of the cell state given the hidden state, the cell state is the "hidden state;" it doesn't appear in the output, but is still represented internally in the loop. 

The basic RNN in the notebook in this directory is all visible state, because the output probabilities are just a function of the full RNN state. To obey the function signature, we just pass a unit given by ``ag_tuple(())`` to the hidden_state field. 

The other arguments to binomial_checkpoint are self explanatory — number of checkpoints to save during reverse-mode AD and number of steps in the loop. 

Calling grad on binomial_checkpoint returns a harcoded derivative which performs the checkpointing algorithm of Gruslys et. al (2016). 

### RNN Script

A demo of binomial_checkpoint applied to a simple RNN is available in this directory as an interactive notebook. Performance benchmarks are computed inline. 

### LSTM Scripts

We can analyze the memory requirements of an LSTM with and without checkpointing by comparing the overhead introduced in each case relative to an identical script with minimized memory requirements. 

First, we can look at the output of top for an uncheckpointed LSTM with sequence length 8. We see Python is
using about 78M of memory. 
```
PID   COMMAND      %CPU  TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    
6129  Python       196.8 00:28.74 2/2   0    15    78M    0B     0B     6129 5987 running  
```

Let's bump up the sequence length to 512 and see what happens. 
```
PID   COMMAND      %CPU  TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    
6136  Python       197.1 00:20.51 2/2   0    15    362M+  0B     0B     6136 5987 running
```

As we would expected, the memory demands are a lot higher. The checkpointed script does better. 
```
PID   COMMAND      %CPU  TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    
6140  Python       141.1 00:48.66 2/1   0    15    163M+  0B     0B     6140 5987 running
```

Per iteration, the checkpointed LSTM takes about 4.60 seconds, while the uncheckpointed LSTM takes 
about 2.05 seconds. The checkpointed LSTM introduces 85 mb of memory over baseline, while the normal
LSTM introduces 284 mb of memory. This corresponds to a slowdown of 2.24x, with memory savings of 3.34x.


