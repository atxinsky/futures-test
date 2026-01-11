
import os

engine_code = open('D:/期货/回测改造/core/etf_backtest_engine_v1_backup.py', 'r', encoding='utf-8').read()

# The new code
new_code = open('D:/temp_engine_v2.txt', 'r', encoding='utf-8').read()

with open('D:/期货/回测改造/core/etf_backtest_engine.py', 'w', encoding='utf-8') as f:
    f.write(new_code)

print('Done')
