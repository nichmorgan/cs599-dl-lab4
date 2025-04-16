"""
BasicLSTM_cell implementation for TensorFlow 2.x
"""

import tensorflow as tf

class BasicLSTM_cell(tf.keras.Model):
    def __init__(self, input_units, hidden_units, output_units):
        """
        Initialize a basic LSTM cell
        
        Args:
            input_units (int): Size of input dimension
            hidden_units (int): Size of hidden state dimension
            output_units (int): Size of output dimension
        """
        super(BasicLSTM_cell, self).__init__()
        
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        
        # Declare weights for input gate
        self.Wi = tf.Variable(tf.zeros([self.input_units, self.hidden_units]), trainable=True, name="Wi")
        self.Ui = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]), trainable=True, name="Ui")
        self.bi = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="bi")
        
        # Declare weights for forget gate
        self.Wf = tf.Variable(tf.zeros([self.input_units, self.hidden_units]), trainable=True, name="Wf")
        self.Uf = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]), trainable=True, name="Uf")
        self.bf = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="bf")
        
        # Declare weights for output gate
        self.Woutg = tf.Variable(tf.zeros([self.input_units, self.hidden_units]), trainable=True, name="Woutg")
        self.Uoutg = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]), trainable=True, name="Uoutg")
        self.boutg = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="boutg")
        
        # Declare weights for cell state
        self.Wc = tf.Variable(tf.zeros([self.input_units, self.hidden_units]), trainable=True, name="Wc")
        self.Uc = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]), trainable=True, name="Uc")
        self.bc = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="bc")

        # Weights for output layers
        self.Wo = tf.Variable(
            tf.random.truncated_normal([self.hidden_units, self.output_units], mean=0, stddev=.02),
            trainable=True, name="Wo"
        )
        self.bo = tf.Variable(
            tf.random.truncated_normal([self.output_units], mean=0, stddev=.02),
            trainable=True, name="bo"
        )

    def initialize_state(self, batch_size):
        """
        Initialize the hidden state and cell state with zeros
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            tuple: (hidden_state, cell_state)
        """
        return (
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units])
        )
    
    def lstm_step(self, previous_state, x):
        """
        Single step of LSTM computation
        
        Args:
            previous_state: Tuple of (previous_hidden_state, previous_cell_state)
            x: Input tensor of shape [batch_size, input_units]
            
        Returns:
            tuple: (current_hidden_state, current_cell_state)
        """
        previous_hidden_state, c_prev = previous_state

        # Input gate
        i = tf.sigmoid(
            tf.matmul(x, self.Wi) +
            tf.matmul(previous_hidden_state, self.Ui) + 
            self.bi
        )

        # Forget gate
        f = tf.sigmoid(
            tf.matmul(x, self.Wf) +
            tf.matmul(previous_hidden_state, self.Uf) + 
            self.bf
        )

        # Output gate
        o = tf.sigmoid(
            tf.matmul(x, self.Woutg) +
            tf.matmul(previous_hidden_state, self.Uoutg) + 
            self.boutg
        )

        # Cell candidate
        c_ = tf.nn.tanh(
            tf.matmul(x, self.Wc) +
            tf.matmul(previous_hidden_state, self.Uc) + 
            self.bc
        )
        
        # Update cell state
        c = f * c_prev + i * c_
        
        # Update hidden state
        current_hidden_state = o * tf.nn.tanh(c)

        return current_hidden_state, c
    
    def call(self, inputs, initial_state=None):
        """
        Process a sequence of inputs
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_length, input_units]
            initial_state: Optional tuple of (initial_hidden_state, initial_cell_state)
            
        Returns:
            tuple: (outputs, (final_hidden_state, final_cell_state))
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        if initial_state is None:
            hidden_state, cell_state = self.initialize_state(batch_size)
        else:
            hidden_state, cell_state = initial_state
        
        all_hidden_states = []
        
        # Process the sequence
        for t in range(seq_length):
            hidden_state, cell_state = self.lstm_step(
                (hidden_state, cell_state), 
                inputs[:, t, :]
            )
            all_hidden_states.append(hidden_state)
        
        # Stack all hidden states
        all_hidden_states = tf.stack(all_hidden_states, axis=1)
        
        # Get outputs by applying output transformation
        outputs = tf.nn.relu(tf.matmul(
            tf.reshape(all_hidden_states, [-1, self.hidden_units]),
            self.Wo
        ) + self.bo)
        
        # Reshape to [batch_size, seq_length, output_units]
        outputs = tf.reshape(outputs, [batch_size, seq_length, self.output_units])
        
        return outputs, (hidden_state, cell_state)