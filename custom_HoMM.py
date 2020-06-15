from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from copy import deepcopy

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import warnings

import default_architecture_config
from utils import save_config


def get_word_embeddings(inputs, vocab_size, dimensionality, reuse=True)
    with tf.variable_scope("word_embeddings", reuse=reuse):
        self.word_embeddings = tf.get_variable(
            "embeddings", shape=[vocab_size, dimensionality])

    embedded_language = tf.nn.embedding_lookup(self.word_embeddings,
                                               language_input)

    return embedded_language


def LSTM_encoder(input_sequence, config, scope="encoder", reuse=True):
    """config should be dict containing dimensionality, num_layers, and seq_length"""
    num_lstm_layers = config["num_layers"]
    with tf.variable_scope(scope, reuse=reuse):
	cells = [tf.nn.rnn_cell.LSTMCell(
	    config["dimensionality"]) for _ in range(num_lstm_layers)]
	stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

	batch_size = tf.shape(language_input)[0]
	state = stacked_cell.zero_state(batch_size, dtype=tf.float32)
    
        processed_outputs = []
	for i in range(config["seq_len"]):
	    this_input = input_sequence[:, i, :]
	    _, state = stacked_cell(this_input, state)

                cell_output = tf.nn.dropout(cell_output,
                                            lang_keep_prob_ph)
                processed_output = slim.fully_connected(
                    cell_output, dimensionality,
                    activation_fn=None)
                processed_outputs.append(processed_output)

    return state, processed_outputs


def input_embeddings_to_LSTM_state(input_embeddings, num_layers, dimensionality,
                                   scope="init_state_construction", reuse=True):
    with tf.variable_scope(scope, reuse=reuse):
        states = []
        for layer_i in range(num_layers):
            this_c = slim.fully_connected(
                         input_embeddings, dimensionality, activation_fn=None)
            this_h = slim.fully_connected(
                               input_embeddings, dimensionality, activation_fn=None)
            states.append(tf.nn.rnn.LSTMStateTuple(this_c, this_h))

    return tuple(states)


def LSTM_decoder(state_input_embeddings, config, scope="decoder", reuse=True):
    """config should be dict containing dimensionality, num_layers, and seq_length"""
    dimensionality = config["dimensionality"]
    num_lstm_layers = config["num_layers"]

    with tf.variable_scope(scope, reuse=reuse):
	cells = [tf.nn.rnn_cell.LSTMCell(
	    dimensionality) for _ in range(num_lstm_layers)]
	stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

	state = input_embeddings_to_LSTM_state(
            state_input_embeddings, num_lstm_layers, dimensionality, reuse=reuse)

	this_input = tf.zeros([batch_size, dimensionality])
	output_logits = []
	for i in range(self.config["seq_len"]):
	    cell_output, state = stacked_cell(this_input, state)

	    this_output_logits = slim.fully_connected(
		cell_output, self.vocab_size,
		activation_fn=internal_nonlinearity)
	    output_logits.append(this_output_logits)

	    greedy_output = tf.argmax(this_output_logits, axis=-1)
	    this_input = cell_output

    output_logits = tf.stack(output_logits, axis=1)
    return output_logits


class arithmetic_HoMM(object):
    def __init__(self, config):
        self.config = config
        self._build_architecture()
        self._sess_and_init()
        self.current_lr = config["init_learning_rate"]

        self.output_filename = None
        if config["output_dir"] is not None:
            output_dir = config["output_dir"]
            filename_prefix = config["filename_prefix"]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.output_filename = output_dir + filename_prefix + "losses.csv"
            config_filename = output_dir + filename_prefix + "config.csv"
            save_config(config_filename, config)
   

    def _build_architecture(self):
        internal_nonlinearity = tf.nn.leaky_relu
        self.vocab_size = config["num_symbols"]
        dimensionality = self.config["dimensionality"]

        #### Shared placeholders
        self.task_ph = tf.placeholder(
            tf.int32)  # only one task at a time
        self.input_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["in_seq_len"]])
        self.lr_ph = tf.placeholder(tf.float32)
        
        #### Evaluation (direct) placeholders
        self.eval_target_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["out_seq_len"]])
        self.eval_mask_ph = tf.placeholder(
            tf.bool, shape=[None, self.config["out_seq_len"]])

        #### Expansion placeholders
        # function inputs output by controller during expansion
        self.expand_input_targets_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["expand_seq_len"],
                             self.config["in_seq_len"]])
        self.expand_input_targets_mask_ph = tf.placeholder(
            tf.bool, shape=[None, self.config["expand_seq_len"],
                            self.config["in_seq_len"]])
        # indices of target function embeddings during expansion
        self.expand_function_targets_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["expand_seq_len"]])  
        self.expand_function_targets_mask_ph = tf.placeholder(
            tf.bool, shape=[None, self.config["expand_seq_len"]])
        # output targets during expansion
        self.expand_output_targets_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["expand_seq_len"],
                             self.config["out_seq_len"]])
        self.expand_output_targets_mask_ph = tf.placeholder(
            tf.bool, shape=[None, self.config["expand_seq_len"],
                            self.config["out_seq_len"]])
            
        #### Meta-placeholders
        self.meta_input_ph = tf.placeholder(
            tf.int32, shape=[None, 1])
        self.meta_target_ph = tf.placeholder(
            tf.int32, shape=[None, 1])
        self.feed_embedding_ph = tf.placeholder(
            tf.float32, shape=[None, dimensionality])

 	
