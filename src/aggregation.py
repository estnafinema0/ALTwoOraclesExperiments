import experiments
import strategies
import database

import numpy as np

import abc
import itertools
from typing import Self, Iterable
import dataclasses
import math


class FilterExpression(abc.ABC):
    @abc.abstractmethod
    def match(self, experiment: experiments.Experiment) -> bool:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def __or__(self, other: 'FilterExpression') -> Self:
        if isinstance(other, FilterExpression):
            return OrExpression(self, other)
        return NotImplemented

    def __and__(self, other: 'FilterExpression') -> Self:
        if isinstance(other, FilterExpression):
            return AndExpression(self, other)
        return NotImplemented

    def __invert__(self) -> Self:
        return NotExpression(self)


class OrExpression(FilterExpression):
    def __init__(self, *expressions: FilterExpression):
        self.expressions = expressions

    def match(self, experiment: experiments.Experiment) -> bool:
        return any(expr.match(experiment) for expr in self.expressions)

    def __repr__(self) -> str:
        return f'OrExpression({", ".join(repr(expr) for expr in self.expressions)})'


class AndExpression(FilterExpression):
    def __init__(self, *expressions: FilterExpression):
        self.expressions = expressions

    def match(self, experiment: experiments.Experiment) -> bool:
        return all(expr.match(experiment) for expr in self.expressions)

    def __repr__(self) -> str:
        return f'AndExpression({", ".join(repr(expr) for expr in self.expressions)})'


class NotExpression(FilterExpression):
    def __init__(self, expression: FilterExpression):
        self.expression = expression

    def match(self, experiment: experiments.Experiment) -> bool:
        return not self.expression.match(experiment)

    def __repr__(self) -> str:
        return f'NotExpression({repr(self.expression)})'


class ExperimentKeyExpression(FilterExpression):
    def __init__(self, key: experiments.Experiments.ExperimentKey):
        self.key = key

    def match(self, experiment: experiments.Experiment) -> bool:
        return self.key.equivalent(experiments.Experiments.ExperimentKey.from_experiment(experiment))

    def __repr__(self) -> str:
        return f'ExperimentKeyExpression({repr(self.key)})'


class ExperimentInExpression(FilterExpression):
    def __init__(self, keys: Iterable[experiments.Experiments.ExperimentKey]):
        self.expressions = tuple(ExperimentKeyExpression(key) for key in keys)

    def match(self, experiment: experiments.Experiment) -> bool:
        return any(expr.match(experiment) for expr in self.expressions)

    def __repr__(self) -> str:
        return f'ExperimentInExpression({self.expressions})'


class ExperimentFilter:
    def __init__(self, expr: FilterExpression):
        self.expr = expr

    def __call__(self, experiment: experiments.Experiment) -> bool:
        return self.expr.match(experiment)

    def __or__(self, other: FilterExpression) -> Self:
        if isinstance(other, FilterExpression):
            return ExperimentFilter(OrExpression(self.expr, other))
        return NotImplemented

    def __and__(self, other: FilterExpression) -> Self:
        if isinstance(other, FilterExpression):
            return ExperimentFilter(AndExpression(self.expr, other))
        return NotImplemented

    def __invert__(self) -> Self:
        return ExperimentFilter(NotExpression(self.expr))


class ExperimentsRange:
    def __init__(self, experiments: Iterable[experiments.Experiment], criterion: ExperimentFilter):
        self.range = [e for e in experiments if criterion(e)]

    def __iter__(self):
        return iter(self.range)

    def __len__(self) -> int:
        return len(self.range)

    def __contains__(self, item: experiments.Experiments.ExperimentKey | experiments.Experiment) -> bool:
        if isinstance(item, experiments.Experiments.ExperimentKey):
            return any(e.key.equivalent(item) for e in self.range)
        elif isinstance(item, experiments.Experiment):
            return any(
                e.key.equivalent(experiments.Experiments.ExperimentKey.from_experiment(item)) for e in self.range
            )
        return False

    def unique_tuples(
        self,
        props: Iterable[str],
    ) -> set[tuple]:
        sel = HistorySelector(*props)
        return {sel.select(next(iter(exp.sorted_histories))) for exp in self}


class ExperimentAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, values: Iterable[tuple]) -> tuple:
        pass


class AverageAggregator(ExperimentAggregator):
    def aggregate(self, values: Iterable[tuple]) -> tuple:
        return tuple(np.average(vals) for vals in zip(*values))


class VarianceAggregator(ExperimentAggregator):
    def aggregate(self, values: Iterable[tuple]) -> tuple:
        return tuple(np.var(vals) for vals in zip(*values))


class BundleAggregator(ExperimentAggregator):
    def aggregate(self, values: Iterable[tuple]) -> tuple:
        return tuple(tuple(vals) for vals in zip(*values))


class PlainCountAggregator(ExperimentAggregator):
    def aggregate(self, values: Iterable[tuple]) -> tuple:
        return (len(tuple(values)),)


class SpreadBundleAggregator(ExperimentAggregator):
    def __init__(self, count: int):
        if count <= 0:
            raise ValueError(f'Count must be integer >= 1, received {count}')
        self.count = count

    def aggregate(self, values: Iterable[tuple]) -> tuple:
        for _ in range(self.count):
            values = itertools.chain.from_iterable(values)
        return tuple(values)


class ComposeAggregator(ExperimentAggregator):
    def __init__(self, *aggregators: ExperimentAggregator):
        if len(aggregators) < 2:
            raise ValueError(f'ComposeAggregator composes mutiple aggrgators, at least 2, not {len(aggregators)}')
        self.aggregators = tuple(reversed(aggregators))

    def aggregate(self, values: Iterable[tuple]) -> tuple:
        result = values
        for agg in self.aggregators:
            result = agg.aggregate(result)
        return result


class MergeBundleAggregator(ExperimentAggregator):
    def __init__(self, layers: int):
        if layers <= 0:
            raise ValueError(f'Layers must be integer >= 1, received {layers}')
        self.layers = layers

    def aggregate(self, values: Iterable[tuple]) -> tuple:
        values = tuple(values)
        for _ in range(self.layers - 1):
            values = itertools.chain.from_iterable(values)
        iterator = iter(values)
        result = list(next(iterator))
        for pack in iterator:
            for i, collection in enumerate(pack):
                result[i] += collection
        return tuple(result)


class ZipAggregator(ExperimentAggregator):
    def aggregate(self, values: Iterable[tuple]) -> tuple:
        return tuple(zip(*values))


# TODO: implement max/min, confidence interval aggregators


class FeatureSelector(abc.ABC):
    @abc.abstractmethod
    def select(self, exp: experiments.Experiment | experiments.ExperimentHistory) -> tuple:
        pass


class HistorySelector(FeatureSelector):
    FIELDS = {field.name for field in dataclasses.fields(experiments.ExperimentHistory)} | {'split', 'budget'}

    def __init__(self, *properties: str):
        if any(prop not in self.FIELDS for prop in properties):
            raise ValueError(f'Invalid properties: {', '.join(prop for prop in properties if prop not in self.FIELDS)}')
        self.properties = properties

    def select(self, exp: experiments.ExperimentHistory) -> tuple:
        return tuple(getattr(exp, prop) for prop in self.properties)


class ExperimentSelector(FeatureSelector):
    def __init__(self, aggregators: Iterable[ExperimentAggregator], *propertiess: Iterable[str] | str, runs: int):
        if not propertiess:
            raise ValueError('At least one property must be provided for selection.')
        if runs <= 0:
            raise ValueError('Runs must be a positive integer.')
        if runs == 1 and (aggregators or type(propertiess[0]) is not str):
            raise ValueError('When runs is 1, properties must be provided as individual strings, not iterables.')
        if runs > 1 and (not aggregators or type(propertiess[0]) is str):
            raise ValueError(
                'When runs is greater than 1, aggregators must be provided and properties must be provided as iterables.'
            )
        self.runs = runs
        self.aggregators = aggregators
        self.propertiess = propertiess

    def select(self, exp: experiments.Experiment) -> tuple:
        if self.runs == 1:
            return self.select_single(exp)
        return self.select_multiple(exp)

    def select_single(self, exp: experiments.Experiment) -> tuple:
        return tuple(getattr(exp.histories[1], prop) for prop in self.propertiess)

    def select_multiple(self, exp: experiments.Experiment) -> tuple:
        if self.runs > exp.runs:
            raise ValueError(f'Experiment has only {exp.runs} runs, but {self.runs} were requested for selection.')
        return tuple(
            aggregator.aggregate(
                HistorySelector(*properties).select(history) for history in exp.sorted_histories[: self.runs]
            )
            for aggregator, properties in zip(self.aggregators, self.propertiess)
        )


class GroupFilter(abc.ABC):
    @abc.abstractmethod
    def group(self, experiment: experiments.Experiment) -> Self:
        pass

    @abc.abstractmethod
    def __eq__(self, other: Self) -> bool:
        pass

    @abc.abstractmethod
    def __ne__(self, other: Self) -> bool:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def printable_key(self) -> tuple:
        pass


class GroupEqFilter(GroupFilter):
    FIELDS = dict(
        **{field.name: field.type for field in dataclasses.fields(experiments.ExperimentHistory)},
        **{
            'dataset': database.CompleteDataset,
            'pool': database.Pool,
            'cold_start_strategy': strategies.ColdStartStrategy,
            'active_learning_strategy': strategies.ActiveLearningStrategy,
            'histories': dict[int, experiments.ExperimentHistory],
        },
        **{'runs': int, 'split': int, 'budget': int},
    )

    def __init__(self, *props: str):
        if any(prop.split('.')[0] not in self.FIELDS for prop in props):
            raise ValueError(
                f'Invalid properties: {', '.join(prop for prop in props if prop.split('.')[0] not in self.FIELDS)}'
            )
        self.props = props
        self.obj = None

    def __eq__(self, value) -> bool:
        if not isinstance(value, GroupFilter):
            return NotImplemented
        if self.obj is None or value.obj is None:
            raise ValueError('GroupFilter object not set for comparison')
        for prop in self.props:
            prop_type = self.FIELDS[prop.split('.')[0]]
            if prop_type == float and not math.isclose(
                self.get_nested(self.obj, prop), self.get_nested(value.obj, prop)
            ):
                return False
            elif self.get_nested(self.obj, prop) != self.get_nested(value.obj, prop):
                return False
        return True

    def __ne__(self, value):
        return not self.__eq__(value)

    def get_nested(self, obj, prop):
        parts = prop.split('.')
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def group(self, experiment: experiments.Experiment) -> Self:
        new = GroupEqFilter(*self.props)
        new.obj = experiment
        return new

    def printable_key(self) -> tuple:
        if self.obj is None:
            raise ValueError('GroupFilter object not set for printable_key')
        return tuple((prop, self.get_nested(self.obj, prop)) for prop in self.props)

    def __repr__(self) -> str:
        return f'GroupEqFilter(props={self.props})'


class GroupRangeFilter(GroupFilter):
    FIELDS = dict(
        **{
            field.name: field.type
            for field in dataclasses.fields(experiments.ExperimentHistory)
            if field.type in (int, float)
        },
        **{'runs': int, 'split': int, 'budget': int},
    )

    def __init__(self, prop: str, start: float, end: float, strict_start: bool = True, strict_end: bool = False):
        if prop not in self.FIELDS:
            raise ValueError(f'Invalid property: {prop}')
        self.prop = prop
        self.start = start
        self.end = end
        self.strict_start = strict_start
        self.strict_end = strict_end
        self.obj = None

    def __eq__(self, value) -> bool:
        if not isinstance(value, GroupFilter):
            return NotImplemented
        if self.obj is None:
            raise ValueError('GroupFilter object not set for comparison')
        val = getattr(self.obj, self.prop)
        if self.strict_start and val <= self.start or not self.strict_start and val < self.start:
            return False
        if self.strict_end and val >= self.end or not self.strict_end and val > self.end:
            return False
        return True

    def __ne__(self, value):
        return not self.__eq__(value)

    def __repr__(self) -> str:
        return f'GroupRangeFilter(prop={self.prop}, range=({'(' if self.strict_start else '['}{self.start}, {self.end}{')' if self.strict_end else ']'})'

    def group(self, experiment: experiments.Experiment) -> Self:
        new = GroupRangeFilter(self.prop, self.start, self.end, self.strict_start, self.strict_end)
        new.obj = experiment
        return new

    def printable_key(self) -> tuple:
        if self.obj is None:
            raise ValueError('GroupFilter object not set for printable_key')
        return (
            (self.prop, getattr(self.obj, self.prop)),
            (
                'range',
                {
                    'start': self.start,
                    'end': self.end,
                    'strict_start': self.strict_start,
                    'strict_end': self.strict_end,
                },
            ),
        )


class ExperimentGroup:
    def __init__(
        self,
        collection: Iterable[Self] | Iterable[experiments.Experiment] | Iterable[experiments.ExperimentHistory],
        aggregators: Iterable[ExperimentAggregator],
        *selectors: ExperimentSelector,
        inner_type: type[Self] | type[experiments.Experiment] | None = None,
        group_keys: Iterable[tuple] | None = None,
    ):
        self.contained = list(collection)
        self.type = type(self.contained[0]) if inner_type is None else inner_type
        self.aggregators = tuple(aggregators)
        if self.type is not experiments.Experiment and selectors:
            raise ValueError('Selectors can only be used for groups of Experiments, not for nested groups.')
        self.selectors = selectors
        self.group_keys = group_keys if group_keys is None else tuple(group_keys)

    @classmethod
    def compose_groups(
        cls,
        rng: ExperimentsRange,
        layers: Iterable[GroupFilter],
        aggregatorss: Iterable[Iterable[ExperimentAggregator]],
        *selectors: FeatureSelector,
    ) -> Self:
        return cls.__compose_groups_rec(rng, layers, aggregatorss, *selectors)

    @staticmethod
    def __compose_groups_rec(
        view: Iterable[experiments.Experiment],
        layers: Iterable[GroupFilter],
        aggregatorss: Iterable[Iterable[ExperimentAggregator]],  # one more than layers
        *selectors: FeatureSelector,
    ) -> Self:
        layers = tuple(layers)
        aggregatorss = tuple(aggregatorss)
        if len(layers) > 0:
            layer, *layers = layers
        else:
            layer = ()
        aggregators, *aggregatorss = aggregatorss
        if not layer:
            return ExperimentGroup(view, aggregators, *selectors, inner_type=experiments.Experiment)

        groups: list[list[GroupFilter]] = []
        filters = []
        for exp in view:
            key = layer.group(exp)
            for i, group in enumerate(groups):
                if next(iter(group)) == key:
                    groups[i].append(key)
                    break
            else:
                groups.append([key])
                filters.append(key.printable_key())

        return ExperimentGroup(
            (
                ExperimentGroup.__compose_groups_rec(
                    list(map(lambda x: x.obj, group)), layers, aggregatorss, *selectors
                )
                for group in groups
            ),
            aggregators,
            inner_type=ExperimentGroup,
            group_keys=filters,
        )

    def to_printable(self):
        if self.group_keys is not None:
            if self.type is not experiments.Experiment:
                return {
                    key: {
                        'aggregated': self.aggregate() if self.aggregators else (),
                        'printable': subgroup.to_printable(),
                    }
                    for key, subgroup in zip(self.group_keys, self.contained)
                }
            else:
                return [
                    {'aggregated': self.aggregate() if self.aggregators else (), 'printable': subgroup.to_printable()}
                    for subgroup in self.contained
                ]

        return self.aggregate()

    def aggregate(self) -> tuple:
        if self.type is experiments.Experiment:
            inner_aggregates = tuple(tuple(selector.select(exp) for exp in self.contained) for selector in self.selectors)  # type: ignore
        else:
            inner_aggregates = tuple(group.aggregate() for group in self.contained)
        return (
            tuple(aggregator.aggregate(inner_aggregates) for aggregator in self.aggregators)
            if self.aggregators
            else BundleAggregator().aggregate(inner_aggregates)
        )

    def unique_tuples(
        self,
        props: Iterable[str],
    ) -> set[tuple]:
        if self.type is experiments.Experiment:
            return ExperimentsRange(self.contained, ExperimentFilter(AndExpression())).unique_tuples(props)

        return set(itertools.chain.from_iterable(subgroup.unique_tuples(props) for subgroup in self.contained))

    def __repr__(self):
        return f'Group(type={self.type.__name__},  aggregators={[type(agg).__name__ for agg in self.aggregators]}, selectors={[type(sel).__name__ for sel in self.selectors]}, contained={self.contained},)'
