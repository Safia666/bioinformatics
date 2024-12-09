from Bio import SeqIO
import pandas as pd
import re
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

#contains the nucleotide sequences of the yeast genome - FASTA file
genome_fasta = "/Users/safiam/Desktop/bioinformatics /Caenorhabditis_elegans.WBcel235.dna_sm.toplevel.fa"
#contains annotations of genes, exons, introns, etc of the yeast genome - GTF file
gtf_file = "/Users/safiam/Desktop/bioinformatics /Caenorhabditis_elegans.WBcel235.113.gtf"

#method to parse the FASTA file
#Source: https://lanadominkovic.medium.com/bioinformatics-101-reading-fasta-files-using-biopython-501c390c6820
def parse_fasta(fasta_path):
    sequences = {}
    with open(fasta_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq) #add each sequence with its ID to sequences
    return sequences

# parse the GTF file
def parse_gtf(gtf_path):
    gtf_columns = ["seqname", "source", "feature", "start", "end", 
        "score", "strand", "frame", "attribute"] 
    rows = [] 
    with open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"): #skips commented lines
                continue
            rows.append(line.strip().split("\t"))
    gtf_df = pd.DataFrame(rows, columns=gtf_columns) #converts rows in the file into a pandas dataframe

    # Extract transcript_id from the 'attribute' column
    gtf_df["transcript_id"] = gtf_df["attribute"].apply(
        lambda x: re.search(r'transcript_id "([^"]+)"', x).group(1) if 'transcript_id "' in x else None
    )
    
    return gtf_df

# get exon and intron information
def extract_exon_intron(gtf_df):
    #filter exons
    exons_df = gtf_df[gtf_df["feature"] == "exon"].copy()

    #convert start and end to int
    exons_df["start"] = exons_df["start"].astype(int)
    exons_df["end"] = exons_df["end"].astype(int)

    #sort exons by sequence and location
    exons_df.sort_values(by=["seqname", "strand", "start"], inplace=True)

    introns = []
    #group by transcript
    grouped = exons_df.groupby(["seqname", "strand", "transcript_id"])
    for (seqname, strand, transcript_id), group in grouped:
        sorted_exons = group.sort_values(by="start")

        #find introns between exons
        if len(sorted_exons) <= 1:
            continue
        for i in range(len(sorted_exons) - 1):
            intron_start = sorted_exons.iloc[i]["end"] + 1
            intron_end = sorted_exons.iloc[i + 1]["start"] - 1
            if intron_start <= intron_end:
                introns.append({
                    "seqname": seqname,
                    "source": "inferred",
                    "feature": "intron",
                    "start": intron_start,
                    "end": intron_end,
                    "strand": strand,
                    "transcript_id": transcript_id
                })

    introns_df = pd.DataFrame(introns)
    return exons_df, introns_df

def label_fasta_sequences(fasta_sequences, exons_df, introns_df):
    labeled_sequences = {}
    for seq_id, sequence in fasta_sequences.items():
        labels = np.full(len(sequence), 'U', dtype='<U1') #default U
        
        #label exons
        seq_exons = exons_df[exons_df["seqname"] == seq_id]
        for _, exon in seq_exons.iterrows():
            start_idx = exon["start"] - 1
            end_idx = exon["end"]  
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(sequence):
                end_idx = len(sequence)
            labels[start_idx:end_idx] = 'E'

        #label introns
        seq_introns = introns_df[introns_df["seqname"] == seq_id]
        for _, intron in seq_introns.iterrows():
            start_idx = intron["start"] - 1
            end_idx = intron["end"]
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(sequence):
                end_idx = len(sequence)
            labels[start_idx:end_idx] = 'I'
            
        labeled_sequences[seq_id] = ''.join(labels)
    return labeled_sequences

# Load sequences
genome_sequences = parse_fasta(genome_fasta)

roman_to_integer = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'X':6
}

#training and testing split
training_data = {seq_id: seq for seq_id, seq in genome_sequences.items() 
                 if roman_to_integer.get(seq_id, 0) in range(1, 3)}

testing_data = {seq_id: seq for seq_id, seq in genome_sequences.items() 
                if roman_to_integer.get(seq_id, 0) in range(3, 4)}

gtf_data = parse_gtf(gtf_file)

training_gtf_data = gtf_data[gtf_data['seqname'].map(roman_to_integer.get).isin(range(1, 4))]
testing_gtf_data = gtf_data[gtf_data['seqname'].map(roman_to_integer.get).isin(range(4, 6))]     

exons, introns = extract_exon_intron(training_gtf_data)
testing_exons, testing_introns = extract_exon_intron(testing_gtf_data)

print(introns.head())
print("Genome Sequences:", list(genome_sequences.keys()))

labeled_sequences = label_fasta_sequences(training_data, exons, introns)
labeled_testing_sequences = label_fasta_sequences(testing_data, testing_exons, testing_introns)

states = ['E', 'I', 'U']
nucleotides = ['A', 'C', 'G', 'T']

emission_counts = {s: Counter() for s in states}
state_counts = Counter()
transition_counts = {s: Counter() for s in states}

for seq_id, seq in training_data.items():
    labels = labeled_sequences[seq_id]
    # Update emission counts
    for i, (nuc, st) in enumerate(zip(seq, labels)):
        if st in states and nuc in nucleotides:
            emission_counts[st][nuc] += 1
            state_counts[st] += 1
    
    # Update transition counts
    for i in range(len(labels) - 1):
        current_state = labels[i]
        next_state = labels[i+1]
        if current_state in states and next_state in states:
            transition_counts[current_state][next_state] += 1

# Compute emission probabilities
emission_probs = {}
for s in states:
    emission_probs[s] = {}
    total = sum(emission_counts[s].values())
    for nuc in nucleotides:
        emission_probs[s][nuc] = emission_counts[s][nuc] / total if total > 0 else 0.25

# Compute transition probabilities
transition_probs = {}
for s in states:
    transition_probs[s] = {}
    total = sum(transition_counts[s].values())
    for s2 in states:
        transition_probs[s][s2] = transition_counts[s][s2] / total if total > 0 else (1.0/len(states))

#initial probabilities
initial_counts = Counter()
num_sequences = len(training_data)
for seq_id, seq in training_data.items():
     start_state = labeled_sequences[seq_id][0]
     initial_counts[start_state] += 1

initial_probs = np.zeros(len(states))
for i, s in enumerate(states):
    initial_probs[i] = initial_counts[s] / num_sequences if num_sequences > 0 else (1.0 / len(states))

#convert probabilities to matrices
transition_mat = np.zeros((len(states), len(states)))
for i, s in enumerate(states):
    for j, s2 in enumerate(states):
        transition_mat[i, j] = transition_probs[s][s2]

emission_mat = np.zeros((len(states), len(nucleotides)))
for i, s in enumerate(states):
    for j, nuc in enumerate(nucleotides):
        emission_mat[i, j] = emission_probs[s][nuc]

print(emission_mat)
print(initial_probs)
print(transition_mat)

# Map states and emissions to indices
state_to_idx = {s: i for i, s in enumerate(states)}
obs_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

# Prepare testing data
encoded_sequence = np.array([[obs_to_idx[nuc]] for seq in testing_data.values() for nuc in seq], dtype=int)

def viterbi(obs, start_prob, trans_prob, emit_prob):
    """
    The Viterbi algorithm computes the most likely sequence of hidden states
    for a given sequence of observations, given the model parameters.

    Parameters:
    -----------
    obs : np.ndarray
        An array of shape (N, 1) containing the observed symbols (in encoded form).
        N is the length of the observation sequence.
    start_prob : np.ndarray
        A 1D array of length S with the initial probabilities of each state.
        S is the number of hidden states.
    trans_prob : np.ndarray
        A 2D array of shape (S, S) containing transition probabilities between states.
        trans_prob[i, j] = P(state_j | state_i)
    emit_prob : np.ndarray
        A 2D array of shape (S, M) containing emission probabilities.
        emit_prob[i, k] = P(observation_k | state_i)
        M is the number of possible observation symbols.

    Returns:
    --------
    path : np.ndarray
        A 1D array of length N giving the most likely state sequence (as state indices).
    """

    # Number of observations
    N = len(obs)
    # Number of states
    S = len(start_prob)

    # delta[i, t]: Probability of the most likely state sequence ending in state i at time t
    # psi[i, t]: The state at time t-1 that leads to the highest delta[i, t]
    delta = np.zeros((S, N))
    psi = np.zeros((S, N), dtype=int)

    # Initialization step: calculate delta for the first observation
    for i in range(S):
        # The probability of starting in state i, emitting the first observation obs[0]
        delta[i, 0] = start_prob[i] * emit_prob[i, obs[0][0]]
        # For the first time step, psi is just 0 as there's no previous state
        psi[i, 0] = 0

    # Recursion step: compute delta and psi for t = 1 to N-1
    for t in range(1, N):
        for j in range(S):
            # For each state j at time t, we look back at all states i at time t-1
            # and choose the one that gives the maximum delta[i, t-1] * trans_prob[i, j]
            max_val = -1.0
            max_state = 0
            for i in range(S):
                # Candidate probability of path: delta at previous time step * transition * emission
                val = delta[i, t-1] * trans_prob[i, j] * emit_prob[j, obs[t][0]]
                # Update max if we found a higher probability path
                if val > max_val:
                    max_val = val
                    max_state = i
            # Store the best found probabilities
            delta[j, t] = max_val
            psi[j, t] = max_state

    # Termination step: find the most likely final state
    # The best final state is the one with the maximum delta at time N-1
    last_state = np.argmax(delta[:, N-1])

    # Backtracking step: reconstruct the state path
    # We know the final state, and we use psi to move backward through time
    path = np.zeros(N, dtype=int)
    path[N-1] = last_state
    for t in range(N-2, -1, -1):
        path[t] = psi[path[t+1], t+1]

    return path

# Run Viterbi on the testing data
hidden_state_sequence = viterbi(encoded_sequence, initial_probs, transition_mat, emission_mat)

idx_to_state = {i: s for s, i in state_to_idx.items()}
decoded_states = [idx_to_state[i] for i in hidden_state_sequence]
decoded_string = ''.join(decoded_states)

true_labels = [label for seq_id in labeled_testing_sequences for label in labeled_testing_sequences[seq_id]]
pred_labels = decoded_states

# Ensure equal length for accuracy calculation
min_len = min(len(true_labels), len(pred_labels))
true_labels = true_labels[:min_len]
pred_labels = pred_labels[:min_len]

acc = accuracy_score(true_labels, pred_labels)
print("Accuracy:", acc)

report = classification_report(true_labels, pred_labels, labels=['E', 'I', 'U'])
print(report)

from collections import Counter

label_counts = Counter(label for seq in labeled_sequences.values() for label in seq)
print("Training label distribution:", label_counts)