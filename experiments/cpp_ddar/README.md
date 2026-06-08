# Experimental C++ DDAR Acceleration

This directory is a first, non-invasive C++ experiment for AlphaGeometry DDAR.
It does **not** replace the running benchmark implementation.

The first target is `dd.match_generic`'s pure substitution backtracking core:

- Python still owns `Graph`, dependency reconstruction, numerical checks, graph
  mutation, and algebraic reasoning.
- C++ receives integer-encoded theorem clauses and cached relation instances,
  then returns the same variable-to-point mappings as `dd.try_to_map`.
- `bench_fast_match_generic.py` compares Python and C++ outputs theorem by
  theorem before reporting timing.

This keeps the equivalence boundary small enough to test before attempting a
larger DDAR/Graph port.
