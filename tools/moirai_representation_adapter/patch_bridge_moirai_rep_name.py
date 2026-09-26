#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[2] / 'causalgap' / 'models' / 'foundation' / 'bridge.py'
s = p.read_text(encoding='utf-8')

old = """    @property
    def representation_hook_name(self):
        return f'{self.model_name}_official_internal_encoder_hidden'
"""

new = """    @property
    def representation_hook_name(self):
        if self.model_name == 'Moirai':
            return 'Moirai_postfusion_prediction_token_pre_param_proj_fixed_average'
        return f'{self.model_name}_official_internal_encoder_hidden'
"""

if old not in s:
    raise SystemExit(
        'Expected representation_hook_name block was not found. bridge.py was NOT modified.'
    )

p.write_text(s.replace(old, new, 1), encoding='utf-8')
print('UPDATED:', p)
