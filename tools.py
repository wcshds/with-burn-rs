import numpy as np


def array_to_str(array: np.ndarray, precision: int = 3):
    shape = array.shape
    str_array = (
        np.array(
            [
                eval(f"""f\"{{each:.{precision}f}}\"""")
                for each in array.flatten().tolist()
            ]
        )
        .reshape(shape)
        .tolist()
    )

    return str(str_array).replace("'", "")
