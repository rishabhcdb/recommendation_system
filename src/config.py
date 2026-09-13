from pathlib import Path
import yaml

class _DotDict(dict):
    

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict):
            return _DotDict(val)
        return val

def _load_config() -> _DotDict:

    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate) as f:
                raw = yaml.safe_load(f)
            return _DotDict(raw)
    raise FileNotFoundError("config.yaml not found in project root.")

cfg = _load_config()
