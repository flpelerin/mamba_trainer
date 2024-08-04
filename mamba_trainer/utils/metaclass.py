class CallableMeta(type):
    def __call__(cls, *args, **kwargs):
        return cls.__call__(*args, **kwargs)


class GlobalsMeta(type):
    def __getattr__(cls, key):
        if key in globals():
            return globals()[key]
        else:
            print(f"Variable '{key}' is undefined in {cls.__name__}, returning None")
            return None

    def __setattr__(cls, key, value):
        globals()[key] = value

class Globals(metaclass=GlobalsMeta):
    pass
