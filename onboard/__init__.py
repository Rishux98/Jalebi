"""Onboard memory mode: compact formats + colibri-style tiered terrain tiles (architecture idea only).

Tiering idea adapted from github.com/JustVugg/colibri (MoE expert offloading: hot experts on GPU, warm in RAM,
cold streamed from disk, learned cache, lookahead prefetch) -> terrain tiles. No colibri code is used.
"""
