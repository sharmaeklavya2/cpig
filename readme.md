# CPIG: Conditional Predicate Implication Graph

A predicate is a function that takes an element of a sample space Ω as input
and returns either `true` or `false`.
Given two predicates f and g, and a set S ⊆ Ω,
we say that 'f implies g conditional on S' if ∀ x ∈ S, f(x) ⟹ g(x).

Let V be a set of predicates on Ω. Let S ⊆ Ω.
Let E\_S = {(u, v): u implies v conditional on S}.
Then the directed graph G = (V, E\_S) is called
a Predicate Implication Graph (PIG) conditional on S.

This program takes a set of predicate implications as a CSV/JSON file
and outputs the PIG in CSV, DOT, SVG, PNG, or PDF format.

Additionally, this program can use trinary logic.
It optionally takes as input a file containing known counterexamples.
Then for any ordered pair (u, v) of predicates, we either know that u implies v,
or we know that u doesn't imply v, or we don't know whether u implies v.

## Input format

An input file contains one entry per implication
(or one entry per counterexample, for the counterexample file).
If the input format is CSV, each entry is a row, except the header row.
If the input format is JSON, then the file contains a list of entries,
where each entry is represented as a dictionary.

Each entry contains the following standard fields:
`from`, `to`, `description`. `description` is optional.
Non-standard fields can also appear for entries, and they help define
the subset of Ω for which an implication is true
(or the subset of Ω for which a counterexample holds true).
