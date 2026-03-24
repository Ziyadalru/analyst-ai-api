import json
import re
import math
import numpy as np

# Regex to replace bare NaN/Infinity tokens that Python's json module emits
# (they're not valid JSON and Safari's JSON.parse rejects them with
# "The string did not match the expected pattern.")
_NAN_RE = re.compile(r'\bNaN\b|\bInfinity\b|-Infinity\b')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def numpy_safe(obj):
    """Round-trip through NumpyEncoder to sanitize np types and NaN/Inf values."""
    raw = json.dumps(obj, cls=NumpyEncoder, allow_nan=True)
    # Replace any remaining bare NaN/Infinity (from Python floats) with null
    clean = _NAN_RE.sub('null', raw)
    return json.loads(clean)
