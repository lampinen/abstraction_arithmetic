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
from utils import save_config, untree_dicts

import arithmetic_for_homm


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

            with tf.variable_scope("encoder_fc", reuse=reuse or i > 0): 
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


            with tf.variable_scope("decoder_fc", reuse=reuse or i > 0): 
                this_output_logits = slim.fully_connected(
                    cell_output, vocab_size,
                    activation_fn=None)
            output_logits.append(this_output_logits)

            greedy_output = tf.argmax(this_output_logits, axis=-1)
            this_input = cell_output

    output_logits = tf.stack(output_logits, axis=1)
    return output_logits


def masked_xe_loss(logits, targets, mask):
    unmasked_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)
    masked_loss = tf.where(mask, unmasked_loss,
                           tf.zeros_like(unmasked_loss))

    return masked_loss


def masked_mse_loss(outputs, targets, mask, reduce_before_masking=True):
    unmasked_loss = tf.square(outputs - targets) 
    if reduce_before_masking:
        unmasked_loss = tf.reduce_sum(unmasked_loss, axis=-1)

    masked_loss = tf.where(mask, unmasked_loss,
                           tf.zeros_like(unmasked_loss))

    return masked_loss


class arithmetic_HoMM(object):
    def __init__(self, config):
        self.config = config
        self._build_architecture()
        self._sess_and_init()
        self.current_lr = config["init_learning_rate"]

        self.output_filename = None
        self.batch_size = config["batch_size"]
        if config["output_dir"] is not None:
            output_dir = config["output_dir"]
            self.output_dir = output_dir
            filename_prefix = config["filename_prefix"]
            self.filename_prefix = filename_prefix

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
            tf.int32, shape=[None,])  # only one task at a time executing
                                      # but this ph is overloaded for assigning
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
        
        # TODO: think about whether any of these should be unstopped (optionally)
        meta_input_embeddings = tf.stop_gradient(
            tf.nn.embedding_lookup(self.task_embeddings,
                                   self.meta_input_ph))

        meta_target_embeddings = tf.stop_gradient(
            tf.nn.embedding_lookup(self.task_embeddings,
                                   self.meta_target_ph))

        exp_fun_targets = tf.stop_gradient(
            tf.nn.embedding_lookup(self.task_embeddings,
                                   self.expand_function_targets_ph))

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

        def expander(task_embedding, processed_input, recurrent_config, scope="expander",
                     reuse=True):
            """config should be dict containing dimensionality, num_layers, and
            seq_length"""
            dimensionality = recurrent_config["dimensionality"]
            num_lstm_layers = recurrent_config["num_layers"]
            seq_len = recurrent_config["seq_len"]
            batch_size = tf.shape(processed_input)[0]

            task_embedding = tf.tile(task_embedding, multiples=[batch_size, 1])
            full_input_embeddings = tf.concat([task_embedding, processed_input],
                                               axis=-1)

            with tf.variable_scope(scope, reuse=reuse):
                cells = [tf.nn.rnn_cell.LSTMCell(
                    dimensionality) for _ in range(num_lstm_layers)]
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

                state = input_embeddings_to_LSTM_state(
                    full_input_embeddings, num_lstm_layers, dimensionality, reuse=reuse)

            this_input = tf.zeros([batch_size, dimensionality])
            step_function_embs = []
            step_input_embs = []
            step_output_embs = []
            for i in range(seq_len):
                with tf.variable_scope(scope, reuse=reuse):
                    cell_output, state = stacked_cell(this_input, state)
                    with tf.variable_scope("function_fc", reuse=reuse or i > 0): 
                        this_step_function_emb = slim.fully_connected(
                            cell_output, dimensionality,
                            activation_fn=None)
                    step_function_embs.append(this_step_function_emb)
                    with tf.variable_scope("input_fc", reuse=reuse or i > 0): 
                        this_step_input_emb = slim.fully_connected(
                            cell_output, dimensionality,
                            activation_fn=None)
                    step_input_embs.append(this_step_input_emb)

                if config["task_conditioned_not_hyper"]:  
                    this_step_output_emb = task_network(this_step_function_emb,
                                                        this_step_input_emb)
                else:
                    this_step_output_emb = tf.map_fn(
                        lambda x: task_network(
                            hyper_network(tf.expand_dims(x[0], axis=0)), 
                            tf.expand_dims(x[1], axis=0)),
                        (this_step_function_emb, this_step_input_emb),
                        dtype=tf.float32)
                    this_step_output_emb = tf.squeeze(this_step_output_emb, axis=1)

                step_output_embs.append(this_step_output_emb)

                this_input = this_step_output_emb 

            step_input_embs = tf.stack(step_input_embs, axis=1)
            step_function_embs = tf.stack(step_function_embs, axis=1)
            step_output_embs = tf.stack(step_output_embs, axis=1)
            return step_input_embs, step_function_embs, step_output_embs

        (expander_input_embs,
         expander_function_embs,
         expander_output_embs) = expander(
            self.curr_task_embedding, self.processed_inputs, 
            recurrent_config={"dimensionality": dimensionality,
                              "num_layers": config["num_expander_layers"],
                              "seq_len": config["expand_seq_len"]},
            reuse=False)

        # decode from expander I/O embs using decoder shared w/ evaluation
        def do_expander_decoding(embs):
           return tf.map_fn(
            lambda emb: LSTM_decoder(
                emb, 
                config={"dimensionality": dimensionality,
                        "num_layers": config["num_decoder_layers"],
                        "vocab_size": self.vocab_size,
                        "seq_len": config["out_seq_len"]},
                scope="decoder"),
            embs)

        expander_input_outputs = do_expander_decoding(expander_input_embs) 
        expander_output_outputs = do_expander_decoding(expander_output_embs) 

        (expander_fed_input_embs,
         expander_fed_function_embs,
         expander_fed_output_embs) = expander(
            self.feed_embedding_ph, self.processed_inputs, 
            recurrent_config={"dimensionality": dimensionality,
                              "num_layers": config["num_expander_layers"],
                              "seq_len": config["expand_seq_len"]})

        expander_fed_input_outputs = do_expander_decoding(
            expander_fed_input_embs) 
        expander_fed_output_outputs = do_expander_decoding(
            expander_fed_output_embs) 


        #### losses + accuracies
        def get_accuracy(outputs, targets, mask):
            """targets should be int"""
            hard_outputs = tf.argmax(outputs, axis=-1, output_type=tf.int32)
            outputs_equal_target = tf.equal(hard_outputs, targets)
            return tf.reduce_mean(
                tf.cast(tf.boolean_mask(outputs_equal_target,
                                        mask),
                        tf.float32))  

        # evaluate
        self.base_eval_loss = masked_xe_loss(logits=self.base_eval_output,
                                             targets=oh_eval_target,
                                             mask=self.eval_mask_ph)

        self.total_base_eval_loss = tf.reduce_mean(self.base_eval_loss)
        self.total_base_eval_accuracy = get_accuracy(
            outputs=self.base_eval_output,
            targets=self.eval_target_ph,
            mask=self.eval_mask_ph)

        self.base_fed_eval_loss = masked_xe_loss(logits=self.base_fed_eval_output,
                                                 targets=oh_eval_target,
                                                 mask=self.eval_mask_ph)

        self.total_base_fed_eval_loss = tf.reduce_mean(self.base_fed_eval_loss)
        self.total_base_fed_eval_accuracy = get_accuracy(
            outputs=self.base_fed_eval_output,
            targets=self.eval_target_ph,
            mask=self.eval_mask_ph)

        # expand

        self.expand_input_loss = masked_xe_loss(logits=expander_input_outputs,
                                                targets=oh_exp_in_target,
                                                mask=self.expand_input_targets_mask_ph)
        self.total_expand_input_loss = tf.reduce_mean(self.expand_input_loss)

        self.expand_output_loss = masked_xe_loss(logits=expander_output_outputs,
                                                 targets=oh_exp_out_target,
                                                 mask=self.expand_output_targets_mask_ph)
        self.total_expand_output_loss = tf.reduce_mean(self.expand_output_loss)
        self.total_expand_output_accuracy = get_accuracy(
            outputs=expander_output_outputs,
            targets=self.expand_output_targets_ph,
            mask=self.expand_output_targets_mask_ph)

        self.expand_fun_loss = masked_mse_loss(outputs=expander_function_embs,
                                               targets=exp_fun_targets,
                                               mask=self.expand_function_targets_mask_ph)
        self.total_expand_fun_loss = tf.reduce_mean(self.expand_fun_loss)

        fun_loss_weight = config["expand_function_loss_weight"]
        self.total_expand_loss = self.total_expand_input_loss + fun_loss_weight * self.total_expand_fun_loss + self.total_expand_output_loss
        self.all_expand_losses = {"expand_total_loss": self.total_expand_loss, 
                                  "expand_input_loss": self.total_expand_input_loss,
                                  "expand_function_loss": self.total_expand_fun_loss,
                                  "expand_output_loss": self.total_expand_output_loss,
                                  "expand_output_accuracy": self.total_expand_output_accuracy}

        self.expand_fed_input_loss = masked_xe_loss(logits=expander_input_outputs,
                                                    targets=oh_exp_in_target,
                                                    mask=self.expand_input_targets_mask_ph)
        self.total_expand_fed_input_loss = tf.reduce_mean(self.expand_fed_input_loss)

        self.expand_fed_output_loss = masked_xe_loss(logits=expander_output_outputs,
                                                     targets=oh_exp_out_target,
                                                     mask=self.expand_output_targets_mask_ph)
        self.total_expand_fed_output_loss = tf.reduce_mean(self.expand_fed_output_loss)

        self.expand_fed_fun_loss = masked_mse_loss(outputs=expander_fed_function_embs,
                                                   targets=exp_fun_targets,
                                                   mask=self.expand_function_targets_mask_ph)
        self.total_expand_fed_fun_loss = tf.reduce_mean(self.expand_fed_fun_loss)
        self.total_expand_fed_loss = self.total_expand_fed_input_loss + fun_loss_weight * self.total_expand_fed_fun_loss + self.total_expand_fed_output_loss

        # meta
        self.meta_map_loss = tf.reduce_mean(
            tf.square(self.meta_map_output_embs - meta_target_embeddings))

        #### optimizer and training

        if self.config["optimizer"] == "Adam":
            self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif self.config["optimizer"] == "RMSProp":
            self.optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        elif self.config["optimizer"] == "SGD":
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        else:
            raise ValueError("Unknown optimizer: %s" % self.config["optimizer"])

        self.evaluate_train_op = self.optimizer.minimize(self.total_base_eval_loss)
        self.expand_train_op = self.optimizer.minimize(self.total_expand_loss)
        self.meta_train_op = self.optimizer.minimize(self.meta_map_loss)

        # for some follow-up experiments, we optimize cached embeddings after
        # guessing them zero-shot, to show the improved efficiency of learning
        # from a good guess at the solution.
        self.optimize_evaluate_op = self.optimizer.minimize(self.total_base_eval_loss,
                                                            var_list=[self.task_embeddings])
        self.optimize_expand_op = self.optimizer.minimize(self.total_expand_loss,
                                                          var_list=[self.task_embeddings])

    def _sess_and_init(self):
        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        if self.config["turn_off_arithmetic_optimization"]:
            off = rewriter_config_pb2.RewriterConfig.OFF
            sess_config.graph_options.rewrite_options.memory_optimization = off
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

    def save_parameters(self, filename):
        self.saver.save(self.sess, filename)

    def restore_parameters(self, filename):
        self.saver.restore(self.sess, filename)

    def build_evaluate_feed_dict(self, inputs, task_id=None,
                                 feed_embedding=None,
                                 targets=None, target_masks=None,
                                 call_type="eval"):

        feed_dict = {}
        feed_dict[self.input_ph] = inputs
        if task_id is None:
            if feed_embedding is None:
                raise ValueError("You must supply a task id or embedding!")
            feed_dict[self.feed_embedding_ph] = feed_embedding 
        else:
            if feed_embedding is not None:
                raise ValueError("You must supply either a task id or embedding, not both.")
            feed_dict[self.task_ph] = np.array([task_id])
        if targets is not None: 
            if target_masks is None:
                raise ValueError("Masks must be provided for targets.")
            feed_dict[self.eval_target_ph] = targets
            feed_dict[self.eval_mask_ph] = target_masks

        if call_type == "train": 
            feed_dict[self.lr_ph] = self.curr_lr

        return feed_dict

    def build_expand_feed_dict(self, inputs, task_id=None, feed_embedding=None,
                               in_targets=None, in_target_masks=None, 
                               fun_targets=None, fun_target_masks=None, 
                               out_targets=None, out_target_masks=None, 
                               call_type="eval"):

        feed_dict = {}
        feed_dict[self.input_ph] = inputs

        if task_id is None:
            if feed_embedding is None:
                raise ValueError("You must supply a task id or embedding!")
            feed_dict[self.feed_embedding_ph] = feed_embedding 
        else:
            if feed_embedding is not None:
                raise ValueError("You must supply either a task id or embedding, not both.")
            feed_dict[self.task_ph] = np.array([task_id])

        targets_and_masks = [in_targets, in_target_masks, out_targets,
                             out_target_masks, fun_targets, fun_target_masks]
        tam_not_none = [x is not None for x in targets_and_masks]
        if any(tam_not_none): 
            if not all(tam_not_none): 
                raise ValueError("All targets and masks must be provided together.")
            feed_dict[self.expand_input_targets_ph] = in_targets
            feed_dict[self.expand_input_targets_mask_ph] = in_target_masks
            feed_dict[self.expand_function_targets_ph] = fun_targets
            feed_dict[self.expand_function_targets_mask_ph] = fun_target_masks
            feed_dict[self.expand_output_targets_ph] = out_targets
            feed_dict[self.expand_output_targets_mask_ph] = out_target_masks

        if call_type == "train": 
            feed_dict[self.lr_ph] = self.curr_lr

        return feed_dict

    def build_meta_feed_dict(self, meta_task_id, input_ids, target_ids=None,
                             call_type="eval"):

        feed_dict = {}
        feed_dict[self.task_ph] = np.array([meta_task_id])
        feed_dict[self.meta_input_ph] = input_ids
        if target_ids is not None: 
            feed_dict[self.meta_target_ph] = target_ids

        if call_type == "train": 
            feed_dict[self.lr_ph] = self.curr_meta_lr

        return feed_dict

    def base_eval(self, fun_dataset):
        results = {"expand": {}, "evaluate": {}}
        # evaluate eval
        batch_size = self.batch_size
        for subset_name, subset in fun_dataset["evaluate"].items():
            results["evaluate"][subset_name] = {"evaluate_loss": 0.,
                                                "evaluate_accuracy": 0.}
            num_points = subset["input"].shape[0]
            num_batches = int(np.ceil(num_points / batch_size))
            evaluate_loss = 0.
            for batch_i in range(num_batches):
                feed_dict = self.build_evaluate_feed_dict(
                    inputs=subset["input"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    task_id=subset["task"],
                    targets=subset["eval_target"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    target_masks=subset["eval_mask"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    call_type="eval")  
                this_losses = self.sess.run([self.total_base_eval_loss, self.total_base_eval_accuracy],
                                            feed_dict=feed_dict)
                results["evaluate"][subset_name]["evaluate_loss"] += this_losses[0]
                results["evaluate"][subset_name]["evaluate_accuracy"] += this_losses[1]
            for (k, v) in results["evaluate"][subset_name].items():
                results["evaluate"][subset_name][k] = v / num_batches
                 
        # expand eval
        if fun_dataset["expand"] is not None:
            for subset_name, subset in fun_dataset["expand"].items():
                results["expand"][subset_name] = {k: 0. for k in self.all_expand_losses.keys()}
                num_points = subset["input"].shape[0]
                num_batches = int(np.ceil(num_points / batch_size))
                if num_batches == 0:
                    print("Skipping: {}, {}".format(subset_name, subset))
                    continue
                expand_loss = 0.
                for batch_i in range(num_batches):
                    feed_dict = self.build_expand_feed_dict(
                        inputs=subset["input"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        task_id=subset["task"],
                        in_targets=subset["exp_in_targs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        in_target_masks=subset["exp_in_targ_masks"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        fun_targets=subset["exp_fun_targs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        fun_target_masks=subset["exp_fun_targ_masks"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        out_targets=subset["exp_out_targs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        out_target_masks=subset["exp_out_targ_masks"][batch_i * batch_size:(batch_i + 1) * batch_size],
                        call_type="eval")  
                    this_losses = self.sess.run(self.all_expand_losses,
                                                feed_dict=feed_dict)
                    for k in self.all_expand_losses.keys():
                        results["expand"][subset_name][k] += this_losses[k]
                for (k, v) in results["expand"][subset_name].items():
                    results["expand"][subset_name][k] = v / num_batches
        return results

    def meta_eval(self, meta_dataset):
        results = {}
        for subset_name, subset in meta_dataset.items():
            feed_dict = self.build_meta_feed_dict(
                meta_task_id=subset["task"], 
                input_ids=subset["meta_inputs"], 
                target_ids=subset["meta_targets"],
                call_type="eval")
            results[subset_name] = self.sess.run(self.meta_map_loss,
                                                 feed_dict=feed_dict)
        return results
    
    def do_eval(self, dataset, epoch, output_filename=None):
        if output_filename is None:
            output_filename = self.output_filename
        results = {}
        for fun in self.functions:
            fun_str = arithmetic_for_homm.FUNCTION_STRINGS[fun]
            if fun in self.operations:  # operation 
                results[fun_str] = self.base_eval(dataset[fun])
            else:  # meta
                results[fun_str] = self.meta_eval(dataset[fun])
        results = untree_dicts(results)
        if epoch == 0:
            self.result_keys = list(results.keys())
            self.result_keys.sort()
            self.output_format = "%i, " + ", ".join(["%f"] * len(self.result_keys)) + "\n"
            self.result_keys = ["epoch"] + self.result_keys
            if output_filename is not None:
                with open(output_filename, "w") as fout:
                    fout.write(", ".join(self.result_keys) + "\n")

        results["epoch"] = epoch
        print(results)
        if output_filename is not None:
            with open(output_filename, "a") as fout:
                fout.write(self.output_format % tuple([results[x] for x in self.result_keys]))
        
    def base_train(self, fun_dataset, optimize_task_emb_only=False):
        # evaluate training 
        batch_size = self.batch_size
        subset = fun_dataset["evaluate"]["train"]
        num_points = subset["input"].shape[0]
        if num_points > batch_size:
            batch_indices = np.random.choice(num_points, size=batch_size, replace=False)
            feed_dict = self.build_evaluate_feed_dict(
                inputs=subset["input"][batch_indices],
                task_id=subset["task"],
                targets=subset["eval_target"][batch_indices],
                target_masks=subset["eval_mask"][batch_indices],
                call_type="train")  
        else:
            feed_dict = self.build_evaluate_feed_dict(
                inputs=subset["input"],
                task_id=subset["task"],
                targets=subset["eval_target"],
                target_masks=subset["eval_mask"],
                call_type="train")  

        if optimize_task_emb_only:
            self.sess.run(self.optimize_evaluate_op,
                          feed_dict=feed_dict)
        else:
            self.sess.run(self.evaluate_train_op,
                          feed_dict=feed_dict)
                 
        # expand training
        if fun_dataset["expand"] is not None:
            subset = fun_dataset["expand"]["train"]
            num_points = subset["input"].shape[0]
            if num_points > batch_size:
                batch_indices = np.random.choice(num_points, size=batch_size, replace=False)
                feed_dict = self.build_expand_feed_dict(
                    inputs=subset["input"][batch_indices],
                    task_id=subset["task"],
                    in_targets=subset["exp_in_targs"][batch_indices],
                    in_target_masks=subset["exp_in_targ_masks"][batch_indices],
                    fun_targets=subset["exp_fun_targs"][batch_indices],
                    fun_target_masks=subset["exp_fun_targ_masks"][batch_indices],
                    out_targets=subset["exp_out_targs"][batch_indices],
                    out_target_masks=subset["exp_out_targ_masks"][batch_indices],
                    call_type="train")  
            else:
                feed_dict = self.build_expand_feed_dict(
                    inputs=subset["input"],
                    task_id=subset["task"],
                    in_targets=subset["exp_in_targs"],
                    in_target_masks=subset["exp_in_targ_masks"],
                    fun_targets=subset["exp_fun_targs"],
                    fun_target_masks=subset["exp_fun_targ_masks"],
                    out_targets=subset["exp_out_targs"],
                    out_target_masks=subset["exp_out_targ_masks"],
                    call_type="train")  
            if optimize_task_emb_only:
                self.sess.run(self.optimize_expand_op,
                              feed_dict=feed_dict)
            else:
                self.sess.run(self.expand_train_op,
                              feed_dict=feed_dict)

    def meta_train(self, meta_dataset):
        subset=meta_dataset["train"]
        feed_dict = self.build_meta_feed_dict(
            meta_task_id=subset["task"],
            input_ids=subset["meta_inputs"],
            target_ids=subset["meta_targets"],
            call_type="train")
        self.sess.run(self.meta_train_op,
                      feed_dict=feed_dict)

    def _end_epoch_calls(self, epoch):
        if epoch % self.config["lr_decays_every"] == 0:
            if self.curr_lr > self.config["min_learning_rate"]:
                self.curr_lr *= self.config["lr_decay"]
            if self.curr_meta_lr > self.config["min_meta_learning_rate"]:
                self.curr_meta_lr *= self.config["meta_lr_decay"]

    def initialize_training(self, dataset):
        self.operations = dataset["operations"]
        self.functions = dataset["functions"]
        self.meta_mappings = [x for x in self.functions if x not in self.operations]
        self.curr_lr = self.config["init_learning_rate"]
        self.curr_meta_lr = self.config["init_meta_learning_rate"]

    def run_training(self, dataset, functions_to_skip=[], initialize_training=True):
        if initialize_training:
            self.initialize_training(dataset)
        eval_every = self.config["eval_every"]
        train_meta = self.config["train_meta"]

        self.do_eval(dataset, epoch=0)
        for epoch_i in range(1, self.config["num_epochs"] + 1):
            for fun in self.functions:
                if fun in functions_to_skip:  # for curriculum purposes, etc
                    continue
                if fun in self.operations:  # operation 
                        self.base_train(dataset[fun])
                else:  # meta
                    if train_meta:
                        self.meta_train(dataset[fun])
            if epoch_i % eval_every == 0:
                self.do_eval(dataset, epoch=epoch_i)
            self._end_epoch_calls(epoch_i)

    def guess_embeddings(self, dataset, guess_type="meta_mapping"):
        # update base embedding for eval tasks with meta-mapped versions
        update_inds = []
        update_values = []
        centroid = None
        for meta_mapping in self.meta_mappings:
            meta_dataset = dataset[meta_mapping] 
            if guess_type == "centroid":
                subset = meta_dataset["train"]  # update new task embeddings
                input_ids = np.concatenate([subset["meta_inputs"],
                                            subset["meta_targets"]], axis=0)
                input_ids = np.array(list(set(np.squeeze(input_ids))))
                feed_dict = self.build_meta_feed_dict(
                    meta_task_id=subset["task"],
                    input_ids=subset["meta_inputs"],
                    call_type="eval")

                result_embeddings = self.sess.run(self.curr_task_embedding,
                                                  feed_dict=feed_dict)
                centroid = np.mean(result_embeddings, axis=0, keepdims=True)
            subset = meta_dataset["test"]  # update new task embeddings
            target_ids = subset["meta_targets"]
            if guess_type == "meta_mapping":

                feed_dict = self.build_meta_feed_dict(
                    meta_task_id=subset["task"],
                    input_ids=subset["meta_inputs"],
                    call_type="eval")

                result_embeddings = self.sess.run(self.meta_map_output_embs,
                                                  feed_dict=feed_dict)
            elif guess_type == "random":
                dimensionality = self.config["dimensionality"]
                scale = 1./ np.sqrt(dimensionality) 
                result_embeddings = scale * np.random.normal(size=[len(target_ids),
                                                               dimensionality]) 
            elif guess_type == "centroid":
                result_embeddings = np.tile(centroid, [len(target_ids), 1])

            target_ids = np.squeeze(target_ids, axis=-1)

            if len(result_embeddings.shape) > 2:
                result_embeddings = np.squeeze(result_embeddings, axis=1)
            self.sess.run(
                self.update_embeddings,
                feed_dict={
                    self.task_ph: target_ids,
                    self.update_task_embeddings_ph: result_embeddings 
                })
   
    def guess_embeddings_and_optimize(self, dataset, target_functions, 
                                      num_optimization_epochs=1000,
                                      initialize_optimization=True):
        if initialize_optimization:
            self.initialize_training(dataset)

        for guess_type in ["meta_mapping", "random", "centroid"]:
            self.guess_embeddings(dataset=dataset, guess_type=guess_type)

            # set up eval and run
            eval_every = self.config["eval_every"]
            opt_filename = self.output_dir + self.filename_prefix + "guesstype-{}_opt_losses.csv".format(guess_type)

            self.do_eval(dataset, epoch=0, output_filename=opt_filename)
            for epoch in range(1, num_optimization_epochs+1):
                for fun in target_functions:
                    self.base_train(dataset[fun], optimize_task_emb_only=True)

                if epoch % eval_every == 0 or epoch == num_optimization_epochs:
                    self.do_eval(dataset, epoch=epoch, output_filename=opt_filename)
                self._end_epoch_calls(epoch)


if __name__ == "__main__":
    #### config
    run_offset = 2
    num_runs = 3
    condition = "meta_map_curriculum"  # meta_map: learn all but exp with "up" mapping,
                             #           meta-map to exp and optimize exp task
                             #           task embedding
                             # meta_map_curriculum: as above, except full train
                             #                      after meta-mapping
                             # curriculum: as previous, except no meta-mapping,
                             #             just train full after training all
                             #             but exp
                             # untrained: control for meta_map, without initial
                             #            training
                             # train_exp_only: from beginning, learn only exp
    
    turn_off_arithmetic_optimization = False  # if True, avoids a bug in TF
                                              # causing AlreadyExistsError

    #### end config
    if condition in ["train_exp_only", "untrained", "curriculum"]:
        train_meta = False
    else:
        train_meta = True 
    
    for run_i in range(run_offset, run_offset + num_runs):
        print("Running run {} of condition {}".format(run_i, condition))
        np.random.seed(run_i)
        tf.set_random_seed(run_i)
        dataset = arithmetic_for_homm.build_dataset(random_seed=run_i)
        this_config = default_config.default_config
        this_config.update({
            "num_symbols": len(dataset["vocab_dict"]),
            "num_functions": len(dataset["functions"]),
            "in_seq_len": dataset["in_seq_len"],
            "out_seq_len": dataset["out_seq_len"],
            "expand_seq_len": dataset["expand_seq_len"],
            "output_dir": "/mnt/fs4/lampinen/arithmetic_abstraction/with_homm_optimization_faster/",
            "filename_prefix": "condition-{}_run-{}_".format(condition, run_i),
            "turn_off_arithmetic_optimization": turn_off_arithmetic_optimization,
#            "min_learning_rate": 1e-7,
#            "min_meta_learning_rate": 5e-7,
            "train_meta": train_meta
        })
        model = arithmetic_HoMM(config=this_config) 
        if condition == "train_exp_only":
            model.run_training(dataset=dataset, functions_to_skip=[x for x in dataset["functions"] if x != "^"])
            model.save_parameters(this_config["output_dir"] + this_config["filename_prefix"] + "first_phase_parameters")
            continue
        
        if condition not in "untrained":
            model.run_training(dataset=dataset, functions_to_skip=["^"])
        else:
            model.initialize_training(dataset=dataset)

        model.save_parameters(this_config["output_dir"] + this_config["filename_prefix"] + "first_phase_parameters")

        if condition not in ["train_exp_only", "meta_map_curriculum", "curriculum"]:
            model.guess_embeddings_and_optimize(dataset=dataset, target_functions=["^"], initialize_optimization=True)
        elif condition == "meta_map_curriculum":
            model.guess_embeddings(dataset=dataset)

        if condition in ["meta_map_curriculum", "curriculum"]:
            model.run_training(dataset=dataset, functions_to_skip=[], initialize_training=True)
        
        model.save_parameters(this_config["output_dir"] + this_config["filename_prefix"] + "final_parameters")
        tf.reset_default_graph()
