import re

import pandas as pd

escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans('', '', escapes)


def get_extended_name(name):
    re_outer = re.compile(r'([^A-Z ])([A-Z])')
    re_inner = re.compile(r'(?<!^)([A-Z])([^A-Z])')
    return re_outer.sub(r'\1 \2', re_inner.sub(r' \1\2', name))


def clean_docstring(doc):
    return doc.translate(translator).strip()


def geometric_mean(returns: pd.Series) -> float:
    try:
        return (1 + returns).prod()**(1 / len(returns)) - 1
    except ZeroDivisionError:
        return 0
