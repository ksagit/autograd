We can look at the output of top for an uncheckpointed LSTM with sequence length 8. We see Python is
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
LSTM introduces 284 mb of memory. 


