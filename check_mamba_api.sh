python3 -c "
from mamba_ssm import Mamba
import inspect
sig = inspect.signature(Mamba.__init__)
for p in sig.parameters.values():
    if p.name != 'self':
        print(f'{p.name}: default={p.default}')
"
