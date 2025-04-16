"""
GRU cell implementation for TensorFlow 2.x
Based on the paper: 
"On the properties of neural machine translation: Encoder-decoder approaches"
by Cho et al. (2014)
"""

import tensorflow as tf

class GRUCell(tf.keras.Model):
    def __init__(self, input_units, hidden_units, output_units):
        """
        Initialize a GRU cell
        
        Args:
            input_units (int): Size of input dimension
            hidden_units (int): Size of hidden state dimension
            output_units (int): Size of output dimension
        """
        super(GRUCell, self).__init__()
        
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        
        # Update gate weights
        self.Wz = tf.Variable(
            tf.random.truncated_normal(
                [self.input_units + self.hidden_units, self.hidden_units], 
                mean=0, 
                stddev=0.1
            ),
            trainable=True, 
            name="Wz"
        )
        self.bz = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="bz")
        
        # Reset gate weights
        self.Wr = tf.Variable(
            tf.random.truncated_normal(
                [self.input_units + self.hidden_units, self.hidden_units], 
                mean=0, 
                stddev=0.1
            ),
            trainable=True, 
            name="Wr"
        )
        self.br = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="br")
        
        # Candidate activation weights
        self.Ws = tf.Variable(
            tf.random.truncated_normal(
                [self.input_units + self.hidden_units, self.hidden_units], 
                mean=0, 
                stddev=0.1
            ),
            trainable=True, 
            name="Ws"
        )
        self.bs = tf.Variable(tf.zeros([self.hidden_units]), trainable=True, name="bs")
        
        # Output layer weights
        self.Wo = tf.Variable(
            tf.random.truncated_normal(
                [self.hidden_units, self.output_units], 
                mean=0, 
                stddev=0.1
            ),
            trainable=True, 
            name="Wo"
        )
        self.bo = tf.Variable(tf.zeros([self.output_units]), trainable=True, name="bo")
    
    def initialize_state(self, batch_size):
        """
        Initialize hidden state with zeros
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            hidden_state: Initial hidden state
        """
        return tf.zeros([batch_size, self.hidden_units])
    
    def gru_step(self, previous_state, x):
        """
        Single step of GRU computation
        
        Args:
            previous_state: Previous hidden state
            x: Input tensor of shape [batch_size, input_units]
            
        Returns:
            current_state: Updated hidden state
        """
        # Concatenate previous hidden state and input
        combined = tf.concat([previous_state, x], axis=1)
        
        # Update gate
        z_t = tf.sigmoid(tf.matmul(combined, self.Wz) + self.bz)
        
        # Reset gate
        r_t = tf.sigmoid(tf.matmul(combined, self.Wr) + self.br)
        
        # Prepare the reset gate applied to previous state
        reset_state = r_t * previous_state
        reset_combined = tf.concat([reset_state, x], axis=1)
        
        # Candidate activation
        s_tilde = tf.nn.tanh(tf.matmul(reset_combined, self.Ws) + self.bs)
        
        # Final hidden state
        current_state = (1 - z_t) * previous_state + z_t * s_tilde
        
        return current_state
    
    def call(self, inputs, initial_state=None):
        """
        Process a sequence of inputs
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_length, input_units]
            initial_state: Optional initial hidden state
            
        Returns:
            tuple: (outputs, final_state)
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        if initial_state is None:
            hidden_state = self.initialize_state(batch_size)
        else:
            hidden_state = initial_state
        
        all_hidden_states = []
        
        # Process the sequence
        for t in range(seq_length):
            hidden_state = self.gru_step(hidden_state, inputs[:, t, :])
            all_hidden_states.append(hidden_state)
        
        # Stack all hidden states
        all_hidden_states = tf.stack(all_hidden_states, axis=1)
        
        # Apply output layer to all hidden states
        outputs = tf.nn.relu(
            tf.matmul(
                tf.reshape(all_hidden_states, [-1, self.hidden_units]), 
                self.Wo
            ) + self.bo
        )
        
        # Reshape to [batch_size, seq_length, output_units]
        outputs = tf.reshape(outputs, [batch_size, seq_length, self.output_units])
        
        return outputs, hidden_state