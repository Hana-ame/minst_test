Ref: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo

## What is RNN? <a name="whatisRNN"></a>

* Recurrent neural network (RNN) is a type of deep learning model that is mostly used for analysis of sequential data (time series data prediction). 
* There are different application areas that are used: Language model, neural machine translation, music generation, time series prediction, financial prediction, etc. 
* The aim of this implementation is to help to learn structure of basic RNN (RNN cell forward, RNN cell backward, etc..).
* Code is adapted from Andrew Ng's Course 'Sequential models'.

Code: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/BasicRNN


### RNN Cell <a name="RNNCell"></a>

<img width="961" alt="rnn_step_forward" src="https://user-images.githubusercontent.com/10358317/44312581-5a33c700-a403-11e8-968d-a38dd0ab4401.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### RNN Forward Pass <a name="RNNForward"></a>

<img width="811" alt="rnn_fw" src="https://user-images.githubusercontent.com/10358317/44312584-6029a800-a403-11e8-9171-38cb22873bbb.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### RNN Backward Pass <a name="RNNBackward"></a>

<img width="851" alt="rnn_cell_backprop" src="https://user-images.githubusercontent.com/10358317/44312587-661f8900-a403-11e8-831b-2cd7fae23dfb.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### RNN Problem <a name="RNNProblem"></a>
- In theory, RNNs are absolutely capable of handling such “long-term dependencies.” 
- In practice, RNNs don’t seem to be able to learn them. 
- The problem was explored in depth by Hochreiter (1991) [German] and Bengio, et al. (1994) with [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)
