# LLM Candidate Selection (F<sub>θ</sub>)

**(v<sub>t+1</sub>, σ<sub>t</sub>, ρ<sub>t</sub>, η<sub>t</sub>) = F<sub>θ</sub>(x, v<sub>t</sub>, {Γ(u)}<sub>u ∈ C<sub>t</sub>(x)</sub>)**

Prompt definition: [`search.py:1209-1228`](../regtree_agent/search.py#L1209)

This prompt lets the LLM select the next candidate node during hierarchical traversal using the current query, current node, and candidate child summaries.
