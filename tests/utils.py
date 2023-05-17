def type_assert(name, value, expected_type):
    assert isinstance(value, expected_type), (f"Attribute {name} is of type {type(value)}, "
                                              f"expected {expected_type} instead.")


def geq_assert(name, value, min_value):
    assert value >= min_value, f"Attribute {name} is {value}, but has to be >= {min_value}."


def greater_than_assert(name, value, min_value):
    assert value > min_value, f"Attribute {name} is {value}, but has to be > {min_value}."


def eq_assert(name, value, des_value):
    assert value == des_value, f"Attribute {name} is {value}, but has to be {des_value}."


def shape_assert(name, value, des_shape):
    assert value.shape == des_shape, (f"Attribute {name} shape mismatch. Expected {des_shape}, "
                                      f"got {value.shape}.")


def len_assert(name, value, des_len):
    assert len(value) == des_len, (f"Attribute {name} length mismatch. Expected {des_len}, "
                                   f"got {len(value)}.")
