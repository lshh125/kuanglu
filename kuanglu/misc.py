def record_concat(old_rec, new_rec):
    """Concatenate record of losses

    :param old_rec: old record
    :param new_rec: new record
    :return:
    """
    for k in old_rec:
        old_rec[k] += new_rec[k]
    return old_rec