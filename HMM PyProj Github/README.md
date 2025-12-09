# **Analysis of choice sequences using a** **3-state categorical** **hidden Markov model**

Win and go-through-tube choice sequences from test days were encoded as a binary sequence (Win = 1, Go-through-tube = 0). 

For each experimental group with N animals, the binary choice sequences from test days 1-4 (30 trials per day) were concatenated into a single 120-trial sequence for each animal and then pooled to fit a group-level categorical HMM with parameters λ = {T, E, π}, where T is the state transition matrix, E is the emission probability matrix over the two observable choices, and π is the initial state distribution.