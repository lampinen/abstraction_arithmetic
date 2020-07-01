import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

import arithmetic

from utils import save_config

class lstm_seq2seq_model(object):
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
            self.output_filename = output_dir + filename_prefix + "accuracies.csv" 
            config_filename = output_dir + filename_prefix + "config.csv"
            save_config(config_filename, config)

    def _build_architecture(self):
        internal_nonlinearity = tf.nn.leaky_relu
        self.vocab_size = config["num_symbols"] 
        dimensionality = self.config["dimensionality"] 

        self.input_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["in_seq_len"]])
        self.target_ph = tf.placeholder(
            tf.int32, shape=[None, self.config["out_seq_len"]])
        self.target_mask_ph = tf.placeholder(
            tf.bool, shape=[None, self.config["out_seq_len"]])
        self.lr_ph = tf.placeholder(tf.float32)

        one_hot_targets = tf.one_hot(self.target_ph, depth=self.vocab_size)

        with tf.variable_scope("word_embeddings", reuse=False):
            self.word_embeddings = tf.get_variable(
                "embeddings", shape=[self.vocab_size, dimensionality])

        def encoding_network(language_input, reuse=False):
            embedded_language = tf.nn.embedding_lookup(self.word_embeddings,
                                                       language_input)

            num_lstm_layers = self.config["num_layers"] 
            with tf.variable_scope("encoder"):
                cells = [tf.nn.rnn_cell.LSTMCell(
                    dimensionality) for _ in range(num_lstm_layers)]
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

                batch_size = tf.shape(language_input)[0]
                state = stacked_cell.zero_state(batch_size, dtype=tf.float32)

                for i in range(self.config["in_seq_len"]):
                    this_input = embedded_language[:, i, :]
                    _, state = stacked_cell(this_input, state)

            return state 

        self.encoded_input_state = encoding_network(self.input_ph)

        def decoding_network(encoded_input_state, batch_size, reuse=False):
            dimensionality = self.config["dimensionality"] 
            num_lstm_layers = self.config["num_layers"] 
            with tf.variable_scope("decoder"):
                cells = [tf.nn.rnn_cell.LSTMCell(
                    dimensionality) for _ in range(num_lstm_layers)]
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

                state = encoded_input_state 

                this_input = tf.zeros([batch_size, dimensionality])
                output_logits = []
                for i in range(self.config["out_seq_len"]):
#                    this_input = tf.nn.dropout(this_input,
#                                               lang_keep_prob_ph)
                    cell_output, state = stacked_cell(this_input, state)

                    with tf.variable_scope("decoder_fc", reuse=reuse or i > 0):
                        this_output_logits = slim.fully_connected(
                            cell_output, self.vocab_size,
                            activation_fn=internal_nonlinearity)
                    output_logits.append(this_output_logits)

                    greedy_output = tf.argmax(this_output_logits, axis=-1)
                    this_input = tf.nn.embedding_lookup(self.word_embeddings,
                                                        greedy_output)


#                language_hidden = tf.nn.dropout(language_hidden,
#                                                lang_keep_prob_ph)
            output_logits = tf.stack(output_logits, axis=1)
            return output_logits 

        batch_size = tf.shape(self.input_ph)[0]
        self.output_logits = decoding_network(self.encoded_input_state, batch_size)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_targets, logits=self.output_logits)

        hard_outputs = tf.argmax(self.output_logits, axis=-1, output_type=tf.int32)

        outputs_equal_target = tf.equal(hard_outputs, self.target_ph)
        self.total_accuracy = tf.reduce_mean(
            tf.cast(tf.boolean_mask(outputs_equal_target,
                                    self.target_mask_ph), 
                    tf.float32))

        # harder accuracy: number of entirely correct answers 
        self.total_hard_accuracy = tf.reduce_mean(
            tf.map_fn(
                lambda x: tf.cast(tf.reduce_all(tf.boolean_mask(x[0], x[1])), tf.float32), 
                (outputs_equal_target, self.target_mask_ph),
                dtype=tf.float32))

        masked_loss = tf.where(self.target_mask_ph, loss, tf.zeros_like(loss))

        self.total_loss = tf.reduce_mean(masked_loss)

        if self.config["optimizer"] == "Adam":
            self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif self.config["optimizer"] == "RMSProp":
            self.optimizer = tf.train.RMSPropOptimizer(self.lr_ph)
        elif self.config["optimizer"] == "SGD":
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        else:
            raise ValueError("Unknown optimizer: %s" % self.config["optimizer"])

        self.train_op = self.optimizer.minimize(self.total_loss)
    
    def _sess_and_init(self):
        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())


    def build_feed_dict(self, dataset, subset_key=None):
        feed_dict = {}
        if subset_key is not None:
            if isinstance(subset_key, str):
                subset_key = dataset["subsets"][subset_key]

            inputs = dataset["inputs"][subset_key, :]
            targets = dataset["targets"][subset_key, :]
            masks = dataset["masks"][subset_key, :]
        else:
            inputs = dataset["inputs"]
            targets = dataset["targets"]
            masks = dataset["masks"]

        feed_dict[self.input_ph] = inputs
        feed_dict[self.target_ph] = targets
        feed_dict[self.target_mask_ph] = masks
        
        return feed_dict

    def get_accuracy(self, dataset, subset_key=None):
        feed_dict = self.build_feed_dict(dataset, subset_key)
        return self.sess.run([self.total_accuracy, self.total_hard_accuracy], feed_dict=feed_dict)

    def do_eval(self, dataset, epoch):
        results = {}
        this_accuracy = self.get_accuracy(dataset["train"])
        print("Train accuracy {}".format(this_accuracy))
        subset = "train"
        results[subset + "_soft_accuracy"] = this_accuracy[0] 
        results[subset + "_hard_accuracy"] = this_accuracy[1] 
        for (subset, subset_key) in dataset["test"]["subsets"].items():
            this_accuracy = self.get_accuracy(dataset["test"], subset_key)
            print("Test subset: {}, accuracy: {}".format(
                subset,
                this_accuracy))
            results["test_" + subset + "_soft_accuracy"] = this_accuracy[0] 
            results["test_" + subset + "_hard_accuracy"] = this_accuracy[1] 

        # write results
        if self.output_filename is not None:
            if epoch == 0:
                self.output_keys = list(results.keys())
                self.output_keys.sort()
                self.output_keys = ["epoch"] + self.output_keys 
                with open(self.output_filename, "w") as fout:
                    fout.write(", ".join(self.output_keys) + "\n")
                self.output_format = ", ".join(["{}"] * len(self.output_keys)) + "\n" 

            results["epoch"] = epoch
            with open(self.output_filename, "a") as fout:
                fout.write(self.output_format.format(*[results[key] for key in self.output_keys]))
    
    def do_train_epoch(self, dataset):
        num_examples = len(dataset["inputs"])  
        batch_size = self.config["batch_size"]
        
        order = np.random.permutation(num_examples)
        for batch_i in range(int(np.ceil(float(num_examples) / batch_size))):
            this_batch_indices = order[batch_i * batch_size:(batch_i + 1) * batch_size]
            feed_dict = self.build_feed_dict(dataset, subset_key=this_batch_indices)
            feed_dict[self.lr_ph] = self.current_lr
            self.sess.run(self.train_op, feed_dict=feed_dict)

    def _end_epoch_calls(self, epoch):
        if epoch % self.config["lr_decays_every"] == 0:
            if self.current_lr > self.config["min_lr"]:
                self.current_lr *= self.config["lr_decay"]

    def run_training(self, dataset):
        self.do_eval(dataset, epoch=0)

        num_epochs = self.config["num_epochs"]
        for epoch_i in range(1, num_epochs + 1):
            
            self.do_train_epoch(dataset["train"])

            print("Epoch: {}".format(epoch_i))
            self.do_eval(dataset, epoch_i)
            self._end_epoch_calls(epoch_i)


def filter_dataset(dataset, key):
    new_dataset = {}
    for (k, v) in dataset.items():
        if k == "subsets":
            new_dataset[k] = {}
            for (sk, sv) in dataset[k].items():
                new_dataset[k][sk] = sv[key] 
        else:
            new_dataset[k] = v[key] 
    return new_dataset


if __name__ == "__main__":
    condition = "mult_only"
    run_offset = 5
    num_runs = 5
    for run_i in range(run_offset, run_offset + num_runs):
        config = {
            "num_layers": 3,  # number of layers in each lstm
            "dimensionality": 512,
            "batch_size": 10,
            "init_learning_rate": 5e-5,
            "lr_decay": 0.9,
            "lr_decays_every": 20,
            "min_lr": 1e-8,
            "num_epochs": 2000,
            "optimizer": "Adam",
            "output_dir": "/mnt/fs4/lampinen/arithmetic_abstraction/",
            "filename_prefix": "condition_{}_run_{}_".format(condition, run_i)
        }

        dataset = arithmetic.build_dataset()
        if condition != "full_train":
            if condition == "evaluate_only": 
                dataset["train"] = filter_dataset(dataset["train"], dataset["train"]["subsets"]["evaluate"])
            elif condition == "exp_evaluate_only": 
                dataset["train"] = filter_dataset(dataset["train"], dataset["train"]["subsets"]["exponentiation_evaluate"])
            elif condition == "exp_only": 
                dataset["train"] = filter_dataset(dataset["train"], dataset["train"]["subsets"]["exponentiation"])
            elif condition == "mult_evaluate_only": 
                dataset["train"] = filter_dataset(dataset["train"], dataset["train"]["subsets"]["multiplication_evaluate"])
            elif condition == "mult_only": 
                dataset["train"] = filter_dataset(dataset["train"], dataset["train"]["subsets"]["multiplication"])
            else:
                raise ValueError("Unrecognized condition: {}".format(condition))
        num_train = len(dataset["train"]["inputs"])

        config.update({
            "in_seq_len": dataset["in_seq_len"],
            "out_seq_len": dataset["out_seq_len"],
            "num_symbols": len(dataset["vocab_dict"])
        })

        model = lstm_seq2seq_model(config=config) 
        model.run_training(dataset)
        tf.reset_default_graph()

