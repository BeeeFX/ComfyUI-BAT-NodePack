# Vendored for 🦇 Lines (ComfyUI-BAT-NodePack) — not BAT code.
# Origin: https://github.com/xavysp/TEED utils/AF/Fsmish.py + Xsmish.py
# Licence: MIT, (c) 2022 Xavier Soria Poma (see bat_lines_nets/licenses/)
# Changes: the two files merged; @torch.jit.script dropped (same maths, no
# scripting at import).
#
# Smish: Wang, Xueliang, Honge Ren, and Achuan Wang. "Smish: A Novel Activation
# Function for Deep Learning Methods." Electronics 11.4 (2022): 540.
# smish(x) = x * tanh(ln(1 + sigmoid(x)))

import torch
from torch import nn


def smish(input):
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))


class Smish(nn.Module):
    def forward(self, input):
        return smish(input)
