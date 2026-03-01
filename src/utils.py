from abc import ABCMeta
import enum
import inspect


class Constant:
    def __init__(self, value):
        self.value = value

    def __get__(self, *args):
        return self.value

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __getitem__(self, key):
        return self.value[key]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'


class carried_partial_apply:
    def __init__(self, func):
        self.func = func
        self.signature = inspect.signature(func)
        self.has_varargs = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in self.signature.parameters.values())
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise TypeError(
                'Cannot mix positional and keyword arguments in a single call. '
                'Use separate calls for partial application.'
            )

        new_curry = carried_partial_apply(self.func)
        new_curry.args = self.args + args
        new_curry.kwargs = {**self.kwargs, **kwargs}

        if self.has_varargs:
            return new_curry

        bound = False
        try:
            bound_args = new_curry.signature.bind(*new_curry.args, **new_curry.kwargs)
            bound_args.apply_defaults()
            required = [
                p.name
                for p in new_curry.signature.parameters.values()
                if p.default == inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            if all(param in bound_args.arguments for param in required):
                bound = True
        except TypeError:
            pass

        if bound:
            return new_curry.func(*new_curry.args, **new_curry.kwargs)
        else:
            return new_curry

    def call(self, *args, **kwargs):
        all_args = self.args + args
        all_kwargs = {**self.kwargs, **kwargs}
        return self.func(*all_args, **all_kwargs)

    def __repr__(self):
        return f'<carried_partial_apply {self.func.__name__} args={self.args} kwargs={self.kwargs}>'


class EnumABCMeta(ABCMeta, enum.EnumMeta):
    pass
