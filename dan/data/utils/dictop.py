import random
import collections


# random split a dict
def split_dict(data_dict, test_size=0.1):
    
    dict_key = list(data_dict.keys())
    random.shuffle(dict_key)
    train_list = dict_key[int(len(dict_key)*test_size):]
    test_list = dict_key[0:int(len(dict_key)*test_size)]
    train_dict = {}
    for key in train_list:
        train_dict[key] = data_dict[key]
    test_dict = {}
    for key in test_list:
        test_dict[key] = data_dict[key]
    return train_dict, test_dict


def sort_dict(d, sort_key='key'):
    ordered_d = collections.OrderedDict()
    if sort_key == 'key':
        for key in sorted(d.keys()):
            ordered_d[key] = d[key]
    elif sort_key == 'value':
        items = d.items()
        items.sort()
        for key, value in items:
            ordered_d[key] = value
    else:
        raise TypeError('Not supported sort key:{}'.format(sort_key))
    return ordered_d
