import numpy as np


FUNCTIONS = ["n", "s", "+", "*", "^"] + ["up"]
FUNCTION_STRINGS = {  # for writing to files
    "n": "number", 
    "s": "successor",
    "+": "addition",
    "*": "multiplication",
    "^": "exponentiation",
    "up": "up"}
VOCAB =  [str(x) for x in range(10)] + FUNCTIONS + ["<EOS>", "<PAD>"]
VOCAB = {x: i for (i, x) in enumerate(VOCAB)}
FUNCTION_IDS = {x: i for (i, x) in enumerate(FUNCTIONS)}
INV_VOCAB = {i: x for (x, i) in VOCAB.items()}


def encode(x):
    """Note: only for arithmetic strings"""
    if not isinstance(x, str):
        x = str(x)
    return [VOCAB[c] for c in x]


def successor(x):
    return x + 1


def front_pad_and_eos(x, length,
                      pad_token=VOCAB["<PAD>"],
                      eos_token=VOCAB["<EOS>"]):
    return [pad_token] * (length - (len(x) + 1)) + x + [eos_token]


def back_pad_and_eos(x, length,
                     pad_token=VOCAB["<PAD>"],
                     eos_token=VOCAB["<EOS>"]):
    return x  + [eos_token] + [pad_token] * (length - (len(x) + 1))


def pad_seq(length, pad_token=VOCAB["<PAD>"]):
    return [pad_token] * length


def arithmetic_expander(arithmetic_string):
    """arithmetic_string should be "x" or "xs" or "x op y" where x, y are ints"""
#    if arithmetic_string[-1] == "s":
#        x = arithmetic_string[:-1]
#        return x + "+1"
    if "+" in arithmetic_string: 
        x, y = arithmetic_string.split("+")
        return x + "s" * int(y)
    elif "*" in arithmetic_string: 
        x, y = arithmetic_string.split("*")
        return "+".join([x] * int(y)) 
    elif "^" in arithmetic_string: 
        x, y = arithmetic_string.split("^")
        y = int(y)
        if y == 0:
            return "1"
        else:
            return "*".join([x] * y) 
    else:  #single number
        return "+".join(["1"] * int(arithmetic_string)) 


def folder(expanded_string):
    """Creates series of partial folds/sub-executions of string of (homogeneous) 
       operations. E.g. folder("3+3+3+3") = [("3+3", "6"), ("6+3", "9"),
       ("9+3", "12")]."""
    folds = []
    if "s" in expanded_string:
       first_ind = expanded_string.index("s") 
       in_val = int(expanded_string[:first_ind])
       for i in range(len(expanded_string) - first_ind): 
           out_val = in_val + 1
           folds.append((str(in_val) + "s",
                         str(out_val)))
           in_val = out_val
    elif "+" in expanded_string or "*" in expanded_string:
        op = "+" if "+" in expanded_string else "*"
        terms = expanded_string.split(op)
        this_out = terms[0]
        for i in range(1, len(terms)):
            this_in = this_out + op + terms[i]
            this_out = str(eval(this_in))
            folds.append((this_in, this_out)) 
    else: 
        raise ValueError("Invalid string passed to folder: {}".format(expanded_string))
    return folds


def build_dataset(max_int=100, 
                  max_rep_int=10,
                  random_seed=0,
                  num_holdouts_per_operation=10,
                  exponentiation_holdouts=["2^5", "4^3", "8^2", "9^1", "0^6", "1^7"]):
    dataset = {}
    dataset["operations"] = ["n", "s", "+", "*", "^"] 
    dataset["functions"] = FUNCTIONS
    dataset["vocab_dict"] = VOCAB
    dataset["inv_vocab_dict"] = INV_VOCAB

    eos_token = VOCAB["<EOS>"]
    pad_token = VOCAB["<PAD>"]

    for operation in dataset["operations"]:
        dataset[operation] = {}
        dataset[operation]["expand"] = None if operation in ["n", "s"] else {
            x: {"task": FUNCTION_IDS[operation], "input": [],
                "exp_in_targs": [], "exp_in_targ_masks": [],
                "exp_fun_targs": [], "exp_fun_targ_masks": [],
                "exp_out_targs": [], "exp_out_targ_masks": []} for x in ["train", "test"]}
        dataset[operation]["evaluate"] = {x: {
            "task": FUNCTION_IDS[operation],
            "input": [], 
            "eval_target": [], "eval_mask": []} for x in ["train", "test"]}
    

    # number naming and successors
    number_holdouts = np.random.choice(max_int, 
                                       size=num_holdouts_per_operation,
                                       replace=False)
    successor_holdouts = np.random.choice(max_int,
                                          size=num_holdouts_per_operation,
                                          replace=False)

    for x in range(max_int): 
        if x in number_holdouts:
            subset = "test"
        else:
            subset = "train"
        dataset["n"]["evaluate"][subset]["input"].append(encode(str(x)))
        dataset["n"]["evaluate"][subset]["eval_target"].append(encode(str(x)))

        if x in successor_holdouts:
            subset = "test"
        else:
            subset = "train"

        if x == max_int - 1:
            continue
        dataset["s"]["evaluate"][subset]["input"].append(encode(str(x)))
        dataset["s"]["evaluate"][subset]["eval_target"].append(encode(str(x + 1)))

    # addition
    possible_problems = {}
    for x in range(max_int):
        for y in range(max_int):
            if x + y >= max_int:
                break
            in_str = str(x) + "+" + str(y)
            possible_problems[in_str] = (x, y)
    addition_holdouts = np.random.choice(list(possible_problems.keys()),
                                         size=num_holdouts_per_operation,
                                         replace=False) 

    for (in_str, (x, y)) in possible_problems.items():
        if in_str in addition_holdouts:
            subset = "test"
        else:
            subset = "train"
        in_enc = encode(in_str)

        dataset["+"]["evaluate"][subset]["input"].append(in_enc) 
        dataset["+"]["evaluate"][subset]["eval_target"].append(encode(str(x+y))) 
        if 0 < y <= max_rep_int:
            dataset["+"]["expand"][subset]["input"].append(in_enc)
            expanded = arithmetic_expander(in_str)
            sub_executions = folder(expanded) 
            in_seq, out_seq = list(zip(*sub_executions)) 
            in_seq = [encode(x) for x in in_seq]
            out_seq = [encode(x) for x in out_seq]
            dataset["+"]["expand"][subset]["exp_in_targs"].append(in_seq)
            dataset["+"]["expand"][subset]["exp_fun_targs"].append([FUNCTION_IDS["s"]] * len(out_seq))
            dataset["+"]["expand"][subset]["exp_out_targs"].append(out_seq)

    # multiplication
    possible_problems = {}
    possible_eval_problems = {}
    for x in range(max_int):
        for y in range(max_int):
            if x * y >= max_int:
                break
            in_str = str(x) + "+" + str(y)
            possible_problems[in_str] = (x, y)
            if x not in [0, 1] and y not in [0, 1]:
                possible_eval_problems[in_str] = (x, y)
    multiplication_holdouts = np.random.choice(list(possible_eval_problems.keys()),
                                               size=num_holdouts_per_operation,
                                               replace=False) 

    for (in_str, (x, y)) in possible_problems.items():
        if in_str in multiplication_holdouts:
            subset = "test"
        else:
            subset = "train"
        in_enc = encode(in_str)

        dataset["*"]["evaluate"][subset]["input"].append(in_enc) 
        dataset["*"]["evaluate"][subset]["eval_target"].append(encode(str(x*y))) 
        if 1 < y <= max_rep_int:
            dataset["*"]["expand"][subset]["input"].append(in_enc)
            expanded = arithmetic_expander(in_str)
            sub_executions = folder(expanded) 
            in_seq, out_seq = list(zip(*sub_executions)) 
            in_seq = [encode(x) for x in in_seq]
            out_seq = [encode(x) for x in out_seq]
            dataset["*"]["expand"][subset]["exp_in_targs"].append(in_seq)
            dataset["*"]["expand"][subset]["exp_fun_targs"].append([FUNCTION_IDS["+"]] * len(out_seq))
            dataset["*"]["expand"][subset]["exp_out_targs"].append(out_seq)

    # exponentiation 
    possible_problems = {}
    possible_eval_problems = {}
    for x in range(max_int):
        for y in range(max_int):
            if x ** y >= max_int or (x <= 1 and y > max_rep_int):
                break
            if x == 0 and y == 0:
                continue
            in_str = str(x) + "^" + str(y)
            possible_problems[in_str] = (x, y)
            if x not in [0, 1] and y not in [0, 1]:
                possible_eval_problems[in_str] = (x, y)

    # keep fixed holdouts if they exist, otherwise sample
    # (small and very skewed set, so fixed is recommended)
    if exponentiation_holdouts is None:  
        exponentiation_holdouts = np.concatenate([
            np.random.choice(list(possible_eval_problems.keys()),
                             size=5,
                             replace=False), 
            np.random.choice(list(possible_problems.keys()),
                             size=num_holdouts_per_operation - 5,
                             replace=False)], 
            axis=0) 

    for (in_str, (x, y)) in possible_problems.items():
        if in_str in exponentiation_holdouts:
            subset = "test"
        else:
            subset = "train"
        in_enc = encode(in_str)

        dataset["^"]["evaluate"][subset]["input"].append(in_enc) 
        dataset["^"]["evaluate"][subset]["eval_target"].append(encode(str(x**y))) 
        if 1 < y <= max_rep_int:
            dataset["^"]["expand"][subset]["input"].append(in_enc)
            expanded = arithmetic_expander(in_str)
            sub_executions = folder(expanded) 
            in_seq, out_seq = list(zip(*sub_executions)) 
            in_seq = [encode(x) for x in in_seq]
            out_seq = [encode(x) for x in out_seq]

            dataset["^"]["expand"][subset]["exp_in_targs"].append(in_seq)
            dataset["^"]["expand"][subset]["exp_fun_targs"].append([FUNCTION_IDS["*"]] * len(out_seq))
            dataset["^"]["expand"][subset]["exp_out_targs"].append(out_seq)

    in_seq_len = 2 * len(str(max_int - 1)) + 2
    out_seq_len = in_seq_len  # have to be equal, because sometimes inputs are outputs 
    dataset["in_seq_len"] = in_seq_len
    dataset["out_seq_len"] = out_seq_len
    dataset["expand_seq_len"] = max_rep_int 

    for op in dataset["operations"]:
        for subset in ["train", "test"]:
            dataset[op]["evaluate"][subset]["input"] = np.array(
                [front_pad_and_eos(x, in_seq_len) for x in dataset[op]["evaluate"][subset]["input"]],
                dtype=np.int32)
            dataset[op]["evaluate"][subset]["eval_target"] = np.array(
                [back_pad_and_eos(x, out_seq_len) for x in dataset[op]["evaluate"][subset]["eval_target"]],
                dtype=np.int32)
            dataset[op]["evaluate"][subset]["eval_mask"] = np.not_equal(
                dataset[op]["evaluate"][subset]["eval_target"], pad_token)
            if op in ["n", "s"]:
                continue
            dataset[op]["expand"][subset]["input"] = np.array(
                [front_pad_and_eos(x, in_seq_len) for x in dataset[op]["expand"][subset]["input"]],
                dtype=np.int32)
            dataset[op]["expand"][subset]["exp_in_targs"] = np.array(
                [[back_pad_and_eos(y, out_seq_len) for y in x] + [pad_seq(out_seq_len)] * (max_rep_int - len(x)) for x in dataset[op]["expand"][subset]["exp_in_targs"]],
                dtype=np.int32)
            dataset[op]["expand"][subset]["exp_in_targ_masks"] = np.not_equal(
                dataset[op]["expand"][subset]["exp_in_targs"], pad_token)
            dataset[op]["expand"][subset]["exp_fun_targs"] = np.array(
                [x + [-1] * (max_rep_int - len(x)) for x in dataset[op]["expand"][subset]["exp_fun_targs"]],
                dtype=np.int32)
            dataset[op]["expand"][subset]["exp_fun_targ_masks"] = np.not_equal(
                dataset[op]["expand"][subset]["exp_fun_targs"], -1)
            dataset[op]["expand"][subset]["exp_out_targs"] = np.array(
                [[back_pad_and_eos(y, out_seq_len) for y in x] + [pad_seq(out_seq_len)] * (max_rep_int - len(x))for x in dataset[op]["expand"][subset]["exp_out_targs"]],
                dtype=np.int32)
            dataset[op]["expand"][subset]["exp_out_targ_masks"] = np.not_equal(
                dataset[op]["expand"][subset]["exp_out_targs"], pad_token)
    

    ## meta up operator
    dataset["up"] = {}
    dataset["up"]["train"] = {"task": FUNCTION_IDS["up"],
                              "meta_inputs": np.array([[FUNCTION_IDS[x]] for x in ["s", "+"]], dtype=np.int32),
                              "meta_targets": np.array([[FUNCTION_IDS[x]] for x in ["+", "*"]], dtype=np.int32)} 
    dataset["up"]["test"] = {"task": FUNCTION_IDS["up"],
                             "meta_inputs": np.array([[FUNCTION_IDS["*"]]], dtype=np.int32),
                             "meta_targets": np.array([[FUNCTION_IDS["^"]]], dtype=np.int32)} 

    return dataset


if __name__ == "__main__":
    print(arithmetic_expander("3+5"))
    print(arithmetic_expander("3*5"))
    print(arithmetic_expander("6^3"))
    print(folder(arithmetic_expander("3^4")))
    x = build_dataset(max_int=1000)
    for op in x["operations"]:
        print()
        print()
        print(op)
        print(x[op])
