import numpy as np
from scipy.optimize import brute

from stratestic.backtesting.combining import StrategyCombiner
from stratestic.utils.exceptions import StrategyRequired, OptimizationParametersInvalid


def strategy_optimizer(strategy_runner, opt_params, runner_args, **kwargs):
    """
    Performs brute force optimization for a trading strategy or a combination of strategies.

    Parameters
    ----------
    strategy_runner : callable
        The function to execute the strategy. It should take the strategy parameters as
        input and return the optimization objective.
    opt_params : tuple
        The ranges for each parameter to be optimized, specified as ((start1, stop1, step1),
        (start2, stop2, step2), ...).
    runner_args : tuple
        Additional arguments required by the `strategy_runner` function.
    **kwargs : dict
        Additional keyword arguments passed to the `brute` function from scipy.optimize.

    Returns
    -------
    ndarray
        The best parameters found during the optimization process.

    Notes
    -----
    This function uses the `brute` method from `scipy.optimize` to explore the parameter space
    defined by `opt_params`. The objective function to be optimized is defined by `strategy_runner`.
    """

    opt = brute(
        strategy_runner, opt_params,
        runner_args,
        finish=None,
        **kwargs
    )

    return opt


def adapt_optimization_input(strategy, params):
    """
    Adapts optimization parameters for use with either a single strategy or a `StrategyCombiner`.

    Parameters
    ----------
    strategy : Strategy or StrategyCombiner
        The strategy or combination of strategies to be optimized.
    params : list, tuple, dict, or np.ndarray
        The optimization parameters. For a single strategy, this should be a dictionary of parameters.
        For a `StrategyCombiner`, this should be a list of dictionaries corresponding to each strategy
        in the combination.

    Returns
    -------
    tuple
        Returns a tuple containing adapted optimization parameters, a mapping of strategy parameters
        (if applicable), and the total number of optimization steps.

    Raises
    ------
    StrategyRequired
        If no strategy is provided.
    OptimizationParametersInvalid
        If the parameters are not in the expected format (e.g., not a list for `StrategyCombiner`
        or not a dictionary for a single strategy).

    Notes
    -----
    This function checks the type of the input strategy and parameters, ensuring they are in
    the correct format for optimization. It raises appropriate exceptions for invalid inputs.
    """

    if not strategy:
        raise StrategyRequired

    if isinstance(strategy, StrategyCombiner):
        if not isinstance(params, (list, tuple, type(np.array([])))):
            raise OptimizationParametersInvalid('Optimization parameters must be provided as a list'
                                                ' of dictionaries with the parameters for each individual strategy')

        if len(params) != len(strategy.strategies):
            raise OptimizationParametersInvalid(f'Wrong number of parameters. '
                                                f'Number of strategies is {len(strategy.strategies)}')

        opt_params = []
        strategy_params_mapping = []
        optimization_steps = 1
        for i, strategy in enumerate(strategy.strategies):
            strategy_params, opt_steps = _get_optimization_input(params[i], strategy)
            opt_params.extend(strategy_params)
            strategy_params_mapping.append(len(strategy_params))

            optimization_steps *= opt_steps

        return opt_params, strategy_params_mapping, optimization_steps

    else:
        if not isinstance(params, dict):
            raise OptimizationParametersInvalid('Optimization parameters must be provided as a '
                                                'dictionary with the parameters the strategy')

        strategy_params, optimization_steps = _get_optimization_input(params, strategy)

        return strategy_params, None, optimization_steps


def _get_optimization_input(optimization_params, strategy):
    """
    Internal function to prepare optimization parameters for a single strategy.

    Parameters
    ----------
    optimization_params : dict
        The optimization parameters specified by the user for the strategy.
    strategy : Strategy
        The strategy object for which optimization parameters are being prepared.

    Returns
    -------
    tuple
        A tuple containing the parameters ready for optimization and the total number of optimization steps.

    Notes
    -----
    This function converts user-specified optimization parameters into a format suitable
    for the brute force optimization method. It calculates the total number of optimization
    steps based on the parameter ranges and steps.
    """

    opt_params = []
    optimizations_steps = 1
    for param in strategy.params:

        if param not in optimization_params:
            continue

        param_value = getattr(strategy, f"_{param}")
        is_int = isinstance(param_value, int)
        is_float = isinstance(param_value, float)

        step = 1 if is_int else None

        limits = optimization_params[param] \
            if param in optimization_params \
            else (param_value, param_value + 1) if is_int or is_float \
            else None

        if limits is not None:
            params = (*limits, step) if step is not None else limits
            opt_params.append(params)

            optimizations_steps *= (limits[1] - limits[0])

    return opt_params, optimizations_steps


def get_params_mapping(strategy, parameters, strategy_params_mapping, optimization_params):
    """
    Maps optimized parameters back to strategy parameters for updating or evaluation.

    Parameters
    ----------
    strategy : Strategy or StrategyCombiner
        The strategy or combination of strategies being optimized.
    parameters : ndarray or list
        The optimized parameters.
    strategy_params_mapping : list or None
        A list indicating the mapping of parameters to strategies in a `StrategyCombiner`.
        None if the strategy is not a `StrategyCombiner`.
    optimization_params : dict or list of dicts
        The original optimization parameters provided for optimization.

    Returns
    -------
    dict or list of dicts
        The mapped parameters in a format that can be directly used to update the strategy or strategies.

    Notes
    -----
    For a single strategy, this function returns a dictionary of parameters. For a `StrategyCombiner`,
    it returns a list of dictionaries, each corresponding to the parameters for one of the combined strategies.
    """

    if not isinstance(strategy, StrategyCombiner):
        strategy_params = [param for param in strategy.get_params().keys() if param in optimization_params]
        new_params = {strategy_params[i]: parameter for i, parameter in enumerate(parameters)}
    else:
        new_params = []

        j = -1
        for i, mapping in enumerate(strategy_params_mapping):
            params = {}
            strategy_params = [param for param in strategy.get_params(strategy_index=i).keys()
                               if param in optimization_params[i]]
            for k, j in enumerate(range(j + 1, j + 1 + mapping)):
                params.update({strategy_params[k]: parameters[j]})

            new_params.append(params)

    return new_params
