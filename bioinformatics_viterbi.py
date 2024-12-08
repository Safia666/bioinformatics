from Bio import SeqIO
import pandas as pd
import re
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter

#contains the nucleotide sequences of the yeast genome - FASTA file
genome_fasta = "Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"
#contains annotations of genes, exons, introns, etc of the yeast genome - GTF file
gtf_file = "Saccharomyces_cerevisiae.R64-1-1.113.gtf"

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
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14,
    'XV': 15, 'XVI': 16
}

#training and testing split
training_data = {seq_id: seq for seq_id, seq in genome_sequences.items() 
                 if roman_to_integer.get(seq_id, 0) in range(1, 13)}

testing_data = {seq_id: seq for seq_id, seq in genome_sequences.items() 
                if roman_to_integer.get(seq_id, 0) in range(13, 17)}

gtf_data = parse_gtf(gtf_file)

training_gtf_data = gtf_data[gtf_data['seqname'].map(roman_to_integer.get).isin(range(1, 13))]
testing_gtf_data = gtf_data[gtf_data['seqname'].map(roman_to_integer.get).isin(range(13, 17))]     

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
obs_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Prepare testing data
encoded_sequence = np.array([[obs_to_idx[nuc]] for seq in testing_data.values() for nuc in seq], dtype=int)

# Manually implement Viterbi algorithm
def viterbi(obs, start_prob, trans_prob, emit_prob):
    # obs: array of shape (N, 1) with encoded observations
    # start_prob: initial_probs (length S)
    # trans_prob: transition_mat (SxS)
    # emit_prob: emission_mat (SxM)
    
    N = len(obs)
    S = len(start_prob)

    # delta[i, t]: max probability of the most probable path that ends in state i at time t
    # psi[i, t]: argmax of state at time t-1 that leads to i at time t
    delta = np.zeros((S, N))
    psi = np.zeros((S, N), dtype=int)

    # Initialization
    for i in range(S):
        delta[i, 0] = start_prob[i] * emit_prob[i, obs[0][0]]
        psi[i, 0] = 0

    # Recursion
    for t in range(1, N):
        for j in range(S):
            # Compute delta for state j
            max_val = -1
            max_state = 0
            for i in range(S):
                val = delta[i, t-1] * trans_prob[i, j] * emit_prob[j, obs[t][0]]
                if val > max_val:
                    max_val = val
                    max_state = i
            delta[j, t] = max_val
            psi[j, t] = max_state

    # Termination
    path = np.zeros(N, dtype=int)
    last_state = np.argmax(delta[:, N-1])
    path[N-1] = last_state

    # Path backtracking
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

# Ensure equal length (in case of any boundary issue)
min_len = min(len(true_labels), len(pred_labels))
true_labels = true_labels[:min_len]
pred_labels = pred_labels[:min_len]

acc = accuracy_score(true_labels, pred_labels)
print("Accuracy:", acc)
