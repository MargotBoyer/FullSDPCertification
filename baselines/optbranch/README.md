# OptBranch

Anderson, Ma, Li & Sojoudi, *Towards Optimal Branching for Neural Network
Robustness Certification*, arXiv:2101.09306 / JMLR 2025 vol. 26 paper 21-0068.
No official code repository found.

Nothing vendored here: the paper's SDP relaxation (section 2.4) is exactly
this project's `SDP-IP` config, so the baseline is a pure branch-and-bound
wrapper implemented directly against `solve/sdp_solve` — see
`src/baselines_interface/optbranch/README.md` for the full writeup
(method summary, code map, divergences from the paper's own Algorithm 2,
known limitations).
