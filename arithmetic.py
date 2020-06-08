import numpy as np


VOCAB =  [str(x) for x in range(10)] + ["+", "*", "^", "expand", "evaluate", "<EOS>", "<PAD>"]
VOCAB = {x: i for (i, x) in enumerate(VOCAB)}
INV_VOCAB = {i: x for (x, i) in VOCAB.items()}


def encode(x):
    """Note: only for arithmetic strings"""
    if not isinstance(x, str):
        x = str(x)
    return [VOCAB[c] for c in x]


def successor(x):
    return x + 1


def arithmetic_expander(arithmetic_string):
    """arithmetic_string should be "x" or "x op y" where x, y are ints"""
    if "+" in arithmetic_string: 
        x, y = arithmetic_string.split("+")
        return x + "+1" * int(y)
    elif "*" in arithmetic_string: 
        x, y = arithmetic_string.split("*")
        return "+".join([x] * int(y)) 
    elif "^" in arithmetic_string: 
        x, y = arithmetic_string.split("^")
        return "*".join([x] * int(y)) 
    else:  #single number
        return "+".join(["1"] * int(arithmetic_string)) 


def front_pad_and_eos(x, length, 
                      pad_token=VOCAB["<PAD>"],
                      eos_token=VOCAB["<EOS>"]):
    return [pad_token] * (length - (len(x) + 1)) + x + [eos_token]


def back_pad_and_eos(x, length, 
                     pad_token=VOCAB["<PAD>"],
                     eos_token=VOCAB["<EOS>"]):
    return x  + [eos_token] + [pad_token] * (length - (len(x) + 1))



def build_dataset(max_int=100, 
                  max_rep_int=10,
                  holdouts=["34", "42", "89", "71",
                            "2+3", "4+6", "7+5", "9+8", "43+52", "27+19", "56+33", "14+60",
                            "3*4", "5*2", "7*9", "8*6", "5*16", "4*24", "7*11", "2*33",
                            "4^3", "2^5", "8^2", "9^1", "1^11"]):
    dataset = {}
    dataset["operations"] = ["number", "addition", "multiplication", "exponentiation"] 
    dataset["vocab_dict"] = VOCAB
    dataset["inv_vocab_dict"] = INV_VOCAB
    dataset["train"] = {"inputs": [], "targets": [], "operation": [], "evl_or_exp": []}
    dataset["test"] = {"inputs": [], "targets": [], "operation": [], "evl_or_exp": []}

    exp_token = VOCAB["expand"]
    evl_token = VOCAB["evaluate"]
    eos_token = VOCAB["<EOS>"]
    pad_token = VOCAB["<PAD>"]

    for x in range(max_int): 
        if str(x) in holdouts:
            subset = "test"
        else:
            subset = "train"

        dataset[subset]["inputs"].append([evl_token] + encode(str(x)))
        dataset[subset]["targets"].append(encode(str(x)))
        dataset[subset]["operation"].append("number")
        dataset[subset]["evl_or_exp"].append("evaluate")
        if x <= max_rep_int:
            dataset[subset]["inputs"].append([exp_token] + encode(str(x)))
            dataset[subset]["targets"].append(encode(str(x)))
            dataset[subset]["operation"].append("number")
            dataset[subset]["evl_or_exp"].append("expand")

    for x in range(max_int):
        for y in range(max_int):
            if x + y >= max_int:
                break

            in_str = str(x) + "+" + str(y)
            if in_str in holdouts:
                subset = "test"
            else:
                subset = "train"
            in_enc = encode(in_str)

            dataset[subset]["inputs"].append([evl_token] + in_enc)
            dataset[subset]["targets"].append(encode(str(x+y)))
            dataset[subset]["operation"].append("addition")
            dataset[subset]["evl_or_exp"].append("evaluate")
            if y <= max_rep_int:
                dataset[subset]["inputs"].append([exp_token] + in_enc)
                dataset[subset]["targets"].append(encode(arithmetic_expander(in_str)))
                dataset[subset]["operation"].append("addition")
                dataset[subset]["evl_or_exp"].append("expand")

    for x in range(max_int):
        for y in range(max_int):
            if x * y >= max_int:
                break
            in_str = str(x) + "*" + str(y)
            if in_str in holdouts:
                subset = "test"
            else:
                subset = "train"
            in_enc = encode(in_str)

            dataset[subset]["inputs"].append([evl_token] + in_enc)
            dataset[subset]["targets"].append(encode(str(x*y)))
            dataset[subset]["operation"].append("multiplication")
            dataset[subset]["evl_or_exp"].append("evaluate")
            if y <= max_rep_int:
                dataset[subset]["inputs"].append([exp_token] + in_enc)
                dataset[subset]["targets"].append(encode(arithmetic_expander(in_str)))
                dataset[subset]["operation"].append("multiplication")
                dataset[subset]["evl_or_exp"].append("expand")

    for x in range(max_int):
        for y in range(max_int):
            if x ** y >= max_int:
                break
            in_str = str(x) + "^" + str(y)
            if in_str in holdouts:
                subset = "test"
            else:
                subset = "train"
            in_enc = encode(in_str)
            dataset[subset]["inputs"].append([evl_token] + in_enc)
            dataset[subset]["targets"].append(encode(str(x*y)))
            dataset[subset]["operation"].append("exponentiation")
            dataset[subset]["evl_or_exp"].append("evaluate")
            if y <= max_rep_int:
                dataset[subset]["inputs"].append([exp_token] + in_enc)
                dataset[subset]["targets"].append(encode(arithmetic_expander(in_str)))
                dataset[subset]["operation"].append("exponentiation")
                dataset[subset]["evl_or_exp"].append("expand")


    in_seq_len = 1 + max([len(x) for x in dataset["train"]["inputs"] + dataset["test"]["inputs"]]) 
    out_seq_len = 1 + max([len(x) for x in dataset["train"]["targets"] + dataset["test"]["targets"]]) 
    dataset["in_seq_len"] = in_seq_len
    dataset["out_seq_len"] = out_seq_len

    for subset in ["train", "test"]:
        dataset[subset]["inputs"] = np.array([back_pad_and_eos(x, in_seq_len) for x in dataset[subset]["inputs"]],
                                             dtype=np.int32)
        dataset[subset]["targets"] = np.array([back_pad_and_eos(x, out_seq_len) for x in dataset[subset]["targets"]],
                                              dtype=np.int32)
        dataset[subset]["masks"] = np.zeros(
            shape=dataset[subset]["targets"].shape,
            dtype=np.bool)
        for point_i in range(len(dataset[subset]["masks"])):
            dataset[subset]["masks"][point_i, :(np.argmax(np.equal(dataset[subset]["targets"][point_i, :], eos_token)) + 1)] = True

        dataset[subset]["operation"] = np.array(dataset[subset]["operation"])
        dataset[subset]["evl_or_exp"] = np.array(dataset[subset]["evl_or_exp"])

    # test subsets
    dataset["test"]["test_subsets"] = {} 
    for operation in dataset["operations"]:
        operation_key = dataset["test"]["operation"] == operation
        for evl_or_exp in ["evaluate", "expand"]:
            e_o_e_key = dataset["test"]["evl_or_exp"] == evl_or_exp 

            combined = np.logical_and(operation_key, e_o_e_key)
            if np.any(combined):
                dataset["test"]["test_subsets"][operation + "_" + evl_or_exp] = combined

    
    # shuffle train
    train_perm = np.random.permutation(len(dataset["train"]["operation"]))
    dataset["train"]["inputs"] = dataset["train"]["inputs"][train_perm, :] 
    dataset["train"]["targets"] = dataset["train"]["targets"][train_perm, :] 
    dataset["train"]["masks"] = dataset["train"]["masks"][train_perm, :] 
    dataset["train"]["operation"] = dataset["train"]["operation"][train_perm] 
    dataset["train"]["evl_or_exp"] = dataset["train"]["evl_or_exp"][train_perm] 

    # train subsets
    dataset["train"]["train_subsets"] = {} 
    for operation in dataset["operations"]:
        operation_key = dataset["train"]["operation"] == operation
        dataset["train"]["train_subsets"][operation] = operation_key 
        for evl_or_exp in ["evaluate", "expand"]:
            e_o_e_key = dataset["train"]["evl_or_exp"] == evl_or_exp 

            combined = np.logical_and(operation_key, e_o_e_key)
            if np.any(combined):
                dataset["train"]["train_subsets"][operation + "_" + evl_or_exp] = combined
    

    return dataset

if __name__ == "__main__":
    print(arithmetic_expander("3+5"))
    print(arithmetic_expander("3*5"))
    print(arithmetic_expander("3"))
    print(arithmetic_expander("4^2"))
    print(arithmetic_expander("6^3"))
    x = build_dataset()
    print(x)
    print(len(x["train"]["inputs"]))
