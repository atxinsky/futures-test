# coding=utf-8
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print('=' * 60)
    print('Test Optimizer - IF')
    print('=' * 60)

    from optimization import OptunaOptimizer, OptimizationConfig, ParamSpaceManager

    config = OptimizationConfig(
        strategy_name='brother2v6',
        symbols=['IF'],
        train_start='2022-01-01',
        train_end='2024-12-31',
        val_start='2025-01-01',
        val_end='2025-12-31',
        n_trials=30,
        objective='sharpe',
        initial_capital=1000000,
        min_trades=1,
        max_drawdown=0.5
    )

    print('Train: {} ~ {}'.format(config.train_start, config.train_end))
    print('Val: {} ~ {}'.format(config.val_start, config.val_end))

    param_spaces = ParamSpaceManager.get_key_params('brother2v6')
    print('Params: {}'.format(len(param_spaces)))

    optimizer = OptunaOptimizer(config)

    def progress_cb(p, m):
        if int(p * 100) % 25 == 0:
            print('[{:.0f}%] {}'.format(p*100, m))

    optimizer.set_progress_callback(progress_cb)

    result = optimizer.optimize(param_spaces)

    print('')
    print('=' * 60)
    print('Result')
    print('=' * 60)
    print('Best Sharpe: {:.4f}'.format(result.best_value))

    print('')
    print('Best params:')
    for name, value in result.best_params.items():
        print('  {}: {}'.format(name, value))

    if result.train_metrics:
        print('')
        print('Train:')
        print('  Trades: {}'.format(result.train_metrics.get('trades', 0)))
        print('  Return: {:.2f}%'.format(result.train_metrics.get('return', 0)*100))
        print('  Sharpe: {:.4f}'.format(result.train_metrics.get('sharpe', 0)))
        print('  Drawdown: {:.2%}'.format(result.train_metrics.get('drawdown', 0)))

    if result.val_metrics:
        print('')
        print('Validation:')
        print('  Trades: {}'.format(result.val_metrics.get('trades', 0)))
        print('  Return: {:.2f}%'.format(result.val_metrics.get('return', 0)*100))
        print('  Sharpe: {:.4f}'.format(result.val_metrics.get('sharpe', 0)))
        print('  Drawdown: {:.2%}'.format(result.val_metrics.get('drawdown', 0)))


if __name__ == '__main__':
    main()
