import tensorflow as tf

default_config = {
    "num_symbols": 18,
    "num_functions": 6,
    "in_seq_len": 6,
    "expand_seq_len": 10,
    "out_seq_len": 6,
    "dimensionality": 512,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,

    "num_expander_layers": 2,

    "H_num_hidden_layers": 3,  # hyper network hidden layers
    "H_num_hidden": 512,  # hyper network hidden units
    "task_weight_weight_mult": 1.0,  # weight scale init for hyper network to
                                     # approximately preserve variance in task
                                     # network
    
    "F_num_hidden_layers": 3,  # task network num hidden layers
    "F_num_hidden": 64,  # task network hidden units
    
    "F_weight_normalization": False,  # if True, weight vectors in task network
                                      # will be normalized to unit length --
                                      # Jeff Clune says this plus predicting
                                      # magnitude separately helps in this kind
                                      # of context.
    "F_wn_strategy": "standard",  # one of "standard", "unit_until_last", 
                                  # "all_unit", which respectively fit scalar
                                  # weight magnitudes for all, only last layer,
                                  # and none of the task network weights. 
                                  # Use "standard" for the basic weight-norm
                                  # approach.
    
    "task_conditioned_not_hyper": False,  # Control where task net is fixed, but
                                          # receives task rep as an additional
                                          # input, rather than standard HyperNet
                                          # based setup.
    
    "internal_nonlinearity": tf.nn.leaky_relu,  # nonlinearity for hidden layers
                                                # -- note that output to Z is
                                                # linear by default

    "expand_function_loss_weight": 1.,  # how strongly to weight the function
                                        # loss relative to the I/O for expanding

    #### run stuff
    "optimizer": "Adam",  # Adam or RMSProp are supported options
    "num_epochs": 10000,  # number of training epochs
    "eval_every": 10,  # how many epochs between evals

    "batch_size": 20,

    "num_runs": 5, # number of independent runs of the network to complete

    "init_learning_rate": 3e-5,  # initial learning rate for base tasks
    "init_meta_learning_rate": 3e-5,  # for meta-classification and mappings

    "lr_decay": 0.9,  # how fast base task lr decays (multiplicative)
    "meta_lr_decay": 0.9,
    "lr_decays_every": 500,  # lr decays happen once per this many epochs

    "min_learning_rate": 3e-8,  # can't decay past these minimum values 
    "min_meta_learning_rate": 1e-7,
}
