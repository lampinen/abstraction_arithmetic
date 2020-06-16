from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from copy import deepcopy

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import warnings

import default_config
from utils import save_config


def get_word_embeddings(inputs, vocab_size, dimensionality, reuse=True):
    with tf.variable_scope("word_embeddings", reuse=reuse):
        word_embeddings = tf.get_variable(
            "embeddings", shape=[vocab_size, dimensionality])

    embedded_language = tf.nn.embedding_lookup(word_embeddings,
                                               inputs)

    return embedded_language


def LSTM_encoder(input_sequence, config, scope="encoder", reuse=True):
    """config should be dict containing dimensionality, num_layers, and seq_length"""
    num_lstm_layers = config["num_layers"]
    dimensionality = config["dimensionality"]
    with tf.variable_scope(scope, reuse=reuse):
        cells = [tf.nn.rnn_cell.LSTMCell(
            dimensionality) for _ in range(num_lstm_layers)]
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        batch_size = tf.shape(input_sequence)[0]
        state = stacked_cell.zero_state(batch_size, dtype=tf.float32)
    
        processed_outputs = []
        for i in range(config["seq_len"]):
            this_input = input_sequence[:, i, :]
            cell_output, state = stacked_cell(this_input, state)

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
            states.append(tf.nn.rnn_cell.LSTMStateTuple(this_c, this_h))

    return tuple(states)


def LSTM_decoder(state_input_embeddings, config, scope="decoder", reuse=True):
    """config should be dict containing dimensionality, num_layers,
    vocab_size and seq_length"""
    dimensionality = config["dimensionality"]
    num_lstm_layers = config["num_layers"]
    seq_len = config["seq_len"]
    vocab_size = config["vocab_size"] 
    batch_size = tf.shape(state_input_embeddings)[0]

    with tf.variable_scope(scope, reuse=reuse):
        cells = [tf.nn.rnn_cell.LSTMCell(
            dimensionality) for _ in range(num_lstm_layers)]
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        state = input_embeddings_to_LSTM_state(
            state_input_embeddings, num_lstm_layers, dimensionality, reuse=reuse)

        this_input = tf.zeros([batch_size, dimensionality])
        output_logits = []
        for i in range(seq_len):
            cell_output, state = stacked_cell(this_input, state)

            this_output_logits = slim.fully_connected(
                cell_output, vocab_size,
                activation_fn=None)
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
        config = self.config
        internal_nonlinearity = tf.nn.leaky_relu
        self.vocab_size = config["num_symbols"]
        dimensionality = self.config["dimensionality"]

        #### Shared placeholders
        self.task_ph = tf.placeholder(
            tf.int32, shape=[None,])  # only one task at a time
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

         

        #### Placeholder processing
        self.embedded_inputs = get_word_embeddings(
            self.input_ph, self.vocab_size, dimensionality, reuse=False)

        # one-hot targets
        oh_eval_target = tf.one_hot(self.eval_target_ph,
                                       depth=self.vocab_size)

        oh_exp_in_target = tf.one_hot(self.expand_input_targets_ph,
                                      depth=self.vocab_size)
        oh_exp_fun_target = tf.one_hot(self.expand_function_targets_ph,
                                       depth=self.vocab_size)
        oh_exp_out_target = tf.one_hot(self.expand_output_targets_ph,
                                       depth=self.vocab_size)


        #### Language input processing

        self.full_processed_inputs = LSTM_encoder(
            self.embedded_inputs, 
            config={"dimensionality": dimensionality,
                    "num_layers": config["num_encoder_layers"],
                    "seq_len": config["in_seq_len"]},
            scope="input_encoder", reuse=False)

        self.processed_inputs = self.full_processed_inputs[1][-1]  # last output
        print(self.processed_inputs)

        #### Task embeddings
        with tf.variable_scope("task_embeddings"):
            self.task_embeddings = tf.get_variable(
                "cached_task_embeddings",
                [config["num_functions"],
                 dimensionality],
                dtype=tf.float32)

        self.update_task_embeddings_ph = tf.placeholder(
            tf.float32,
            [None, dimensionality])

        self.update_embeddings = tf.scatter_update(
            self.task_embeddings,
            self.task_ph,
            self.update_task_embeddings_ph)

        self.curr_task_embedding = tf.nn.embedding_lookup(self.task_embeddings, 
                                                          self.task_ph)
        
        meta_input_embeddings = tf.nn.embedding_lookup(self.task_embeddings,
                                                       self.meta_input_ph)

        meta_target_embeddings = tf.nn.embedding_lookup(self.task_embeddings,
                                                       self.meta_target_ph)

        #### Hyper network
        F_num_hidden = config["F_num_hidden"]
        F_num_hidden_layers = config["F_num_hidden_layers"]
        
        H_num_hidden = config["H_num_hidden"]
        internal_nonlinearity = config["internal_nonlinearity"]

        tw_range = config["task_weight_weight_mult"]/np.sqrt(
            F_num_hidden * H_num_hidden) # yields a very very roughly O(1) map
        task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                             tw_range)

        def hyper_network(function_embedding, reuse=True):
            with tf.variable_scope('hyper', reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(config["H_num_hidden_layers"]):
                    hyper_hidden = slim.fully_connected(hyper_hidden, H_num_hidden,
                                                        activation_fn=internal_nonlinearity)

                if F_num_hidden_layers == 0:  # linear task network:
                    task_weights = slim.fully_connected(hyper_hidden, dimensionality * dimensionality,
                                                        activation_fn=None,
                                                        weights_initializer=task_weight_gen_init)

                    task_weights = tf.reshape(task_weights, [dimensionality, dimensionality])
                    task_biases = slim.fully_connected(hyper_hidden, dimensionality,
                                                       activation_fn=None)
                    task_biases = tf.squeeze(task_biases, axis=0)
                    hidden_weights = [task_weights]
                    hidden_biases = [task_biases]

                else:
                    hidden_weights = []
                    hidden_biases = []

                    task_weights = slim.fully_connected(hyper_hidden, F_num_hidden*(dimensionality +(F_num_hidden_layers-1)*F_num_hidden + dimensionality),
                                                        activation_fn=None,
                                                        weights_initializer=task_weight_gen_init)

                    task_weights = tf.reshape(task_weights, [-1, F_num_hidden, (dimensionality + (F_num_hidden_layers-1)*F_num_hidden + dimensionality)])
                    task_biases = slim.fully_connected(hyper_hidden, F_num_hidden_layers * F_num_hidden + dimensionality,
                                                       activation_fn=None)

                    Wi = tf.transpose(task_weights[:, :, :dimensionality], perm=[0, 2, 1])
                    bi = task_biases[:, :F_num_hidden]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                    for i in range(1, F_num_hidden_layers):
                        Wi = tf.transpose(task_weights[:, :, dimensionality+(i-1)*F_num_hidden:dimensionality+i*F_num_hidden], perm=[0, 2, 1])
                        bi = task_biases[:, F_num_hidden*i:F_num_hidden*(i+1)]
                        hidden_weights.append(Wi)
                        hidden_biases.append(bi)
                    Wfinal = task_weights[:, :, -dimensionality:]
                    bfinal = task_biases[:, -dimensionality:]

                    for i in range(F_num_hidden_layers):
                        hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                        hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                    Wfinal = tf.squeeze(Wfinal, axis=0)
                    bfinal = tf.squeeze(bfinal, axis=0)
                    hidden_weights.append(Wfinal)
                    hidden_biases.append(bfinal)

                if config["F_weight_normalization"]:
                    def normalize_weights(x):
                        return x / (tf.sqrt(tf.reduce_sum(
                            tf.square(x), axis=0, keepdims=True)) + 1e-6) 
                    hidden_weights = [normalize_weights(x) for x in hidden_weights]


                    F_wn_strategy = config["F_wn_strategy"]
                    if F_wn_strategy == "standard":  
                        # fit scalar magnitudes for each vector, as expressive
                        # as standard, but may be easier to optimize. The
                        # original weight normalization idea
                        task_weight_norms = tf.nn.relu(1. + slim.fully_connected(
                            hyper_hidden, 
                            F_num_hidden_layers*F_num_hidden + dimensionality,
                            activation_fn=None))
                        endpoints = [i * F_num_hidden for i in range(len(hidden_weights))] + [len(hidden_weights) * F_num_hidden + dimensionality]
                        hidden_weights = [tf.multiply(x, task_weight_norms[0, tf.newaxis, endpoints[i]:endpoints[i + 1]]) for (i, x) in enumerate(hidden_weights)]
                    elif F_wn_strategy == "unit_until_last":
                        # leaves lengths as units except last layer 
                        task_weight_norms = tf.nn.relu(1. + slim.fully_connected(
                            hyper_hidden, 
                            dimensionality,
                            activation_fn=None))
                        hidden_weights[-1] = tf.multiply(hidden_weights[-1], task_weight_norms[0, tf.newaxis, :])

                    # all_unit leaves all weight vecs unit length
                    elif F_wn_strategy != "all_unit":   
                        raise ValueError("Unrecognized F_wn_strategy: %s" % F_wn_strategy) 
                        
                return hidden_weights, hidden_biases

        if not config["task_conditioned_not_hyper"]:  
            self.curr_task_params = hyper_network(self.curr_task_embedding,
                                                  reuse=False)
            self.fed_emb_task_params = hyper_network(self.feed_embedding_ph)

        #### task network F: Z -> Z

        if config["task_conditioned_not_hyper"]:  
            # control where instead of hyper network, just condition task net
            # on task representation
            def task_network(task_rep, processed_input, reuse=True):
                with tf.variable_scope('task_net', reuse=reuse):
                    task_reps = tf.tile(task_rep, [tf.shape(processed_input)[0], 1])
                    task_hidden = tf.concat([task_reps, processed_input],
                                            axis=-1)
                    for _ in range(config["F_num_hidden_layers"]):
                        task_hidden = slim.fully_connected(task_hidden, F_num_hidden,
                                                           activation_fn=internal_nonlinearity)

                    raw_output = slim.fully_connected(task_hidden, dimensionality,
                                                      activation_fn=None)

                return raw_output

            self.base_eval_output_emb = task_network(self.curr_task_embedding,
                                                     self.processed_inputs,
                                                     reuse=False)
            self.base_fed_eval_output_emb = task_network(self.feed_embedding_ph,
                                                         self.processed_inputs)
        else: 
            # hyper-network-parameterized rather than fixed + task conditioned
            # (this is the default)
            def task_network(task_params, processed_input):
                hweights, hbiases = task_params
                task_hidden = processed_input
                for i in range(F_num_hidden_layers):
                    task_hidden = internal_nonlinearity(
                        tf.matmul(task_hidden, hweights[i]) + hbiases[i])

                raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

                return raw_output

            self.base_eval_output_emb = task_network(self.curr_task_params,
                                                     self.processed_inputs)
            self.base_fed_eval_output_emb = task_network(self.fed_emb_task_params,
                                                         self.processed_inputs)
            self.meta_map_output_embs = task_network(self.curr_task_params,
                                                     meta_input_embeddings)

        #### language output
        self.base_eval_output = LSTM_decoder(
            self.base_eval_output_emb, 
            config={"dimensionality": dimensionality,
                    "num_layers": config["num_decoder_layers"],
                    "vocab_size": self.vocab_size,
                    "seq_len": config["out_seq_len"]},
            scope="decoder", reuse=False)  
        print(self.base_eval_output)
        self.base_fed_eval_output = LSTM_decoder(
            self.base_fed_eval_output_emb, 
            config={"dimensionality": dimensionality,
                    "num_layers": config["num_decoder_layers"],
                    "vocab_size": self.vocab_size,
                    "seq_len": config["out_seq_len"]},
            scope="decoder")  


        #### Expander core
        #### This recurrent net will take a task embedding and task input,
        #### and decode a sequence of steps each of which consist of sub-task
        #### embeddings and inputs, which are then executed to yield the next
        #### expander input.



        #### losses
        self.base_eval_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=oh_eval_target, logits=self.base_eval_output)

        self.base_eval_masked_loss = tf.where(
            self.eval_mask_ph, 
            self.base_eval_loss, 
            tf.zeros_like(self.base_eval_loss))

        self.total_base_eval_loss = tf.reduce_mean(
            tf.reduce_mean(self.base_eval_masked_loss, axis=-1))

        self.base_fed_eval_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=oh_eval_target, logits=self.base_fed_eval_output)

        self.base_fed_eval_masked_loss = tf.where(
            self.eval_mask_ph, 
            self.base_fed_eval_loss, 
            tf.zeros_like(self.base_fed_eval_loss))

        self.total_base_fed_eval_loss = tf.reduce_mean(
            tf.reduce_mean(self.base_fed_eval_masked_loss, axis=-1))

        # meta
        self.meta_map_loss = tf.reduce_mean(tf.square(self.meta_map_output_embs - meta_target_embeddings))

        #### optimizer and training

        if self.config["optimizer"] == "Adam":
            self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif self.config["optimizer"] == "RMSProp":
            self.optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        elif self.config["optimizer"] == "SGD":
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        else:
            raise ValueError("Unknown optimizer: %s" % self.config["optimizer"])

        #self.train_op = self.optimizer.minimize(self.total_loss)

#    def _sess_and_init(self):
#        # Saver
#        self.saver = tf.train.Saver()
#
#        # initialize
#        sess_config = tf.ConfigProto()
#        sess_config.gpu_options.allow_growth = True
#        self.sess = tf.Session(config=sess_config)
#        self.sess.run(tf.global_variables_initializer())
#
#        save_config(self.config["save_config_filename"],
#                    self.config)




if __name__ == "__main__":
    model = arithmetic_HoMM(config=default_config.default_config) 
