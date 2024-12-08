from Bio import SeqIO
import pandas as pd
import re
import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict

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

# parse the GTF file -- ChatGPT helped me here
def parse_gtf(gtf_path):
    gtf_columns = ["seqname", "source", "feature", "start", "end", 
        "score", "strand", "frame", "attribute"] 
    rows = [] 
    with open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"): #skips commented lines
                continue
            rows.append(line.strip().split("\t"))
    gtf_df = pd.DataFrame(rows, columns=gtf_columns) #converts the rows in the file into pandas dataframe

    # Extract transcript_id from the 'attribute' column to be used for finding introns
    gtf_df["transcript_id"] = gtf_df["attribute"].apply(
        lambda x: re.search(r'transcript_id "([^"]+)"', x).group(1) if 'transcript_id "' in x else None
    )
    
    return gtf_df

# get exon and intron information
# exons are given in the gff file but we have to make inferences for introns based on extrons
def extract_exon_intron(gtf_df):
    #filter exons through feature column
    exons_df = gtf_df[gtf_df["feature"] == "exon"].copy()

    #add start and end value for exons
    exons_df["start"] = exons_df["start"].astype(int)
    exons_df["end"] = exons_df["end"].astype(int)

    #sort exons by sequence and location for processing introns
    exons_df.sort_values(by=["seqname", "strand", "start"], inplace=True)

    introns = []

    #group exons by seqname, strand, and transcript_id to find introns
    # Group by transcript to find introns between exons
    grouped = exons_df.groupby(["seqname", "strand", "transcript_id"])
    for (seqname, strand, transcript_id), group in grouped:
        sorted_exons = group.sort_values(by="start")

        #if there's only one exon, no introns can be inferred
        if len(sorted_exons) <= 1:
            continue

        #identify intron positions between consecutive exons
        for i in range(len(sorted_exons) - 1):
            intron_start = sorted_exons.iloc[i]["end"] + 1
            intron_end = sorted_exons.iloc[i + 1]["start"] - 1

            #ddd the intron if it has a positive length
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
        labels = np.full(len(sequence), 'U', dtype='<U1') #by default, label all as U
        
        #label exons in sequence as 'E'
        seq_exons = exons_df[exons_df["seqname"] == seq_id]
        for _, exon in seq_exons.iterrows():
            start_idx = exon["start"] - 1
            end_idx = exon["end"]  # end is inclusive
            #bound check
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(sequence):
                end_idx = len(sequence)
            labels[start_idx:end_idx] = 'E'

        #label introns in sequence as 'I'
        seq_introns = introns_df[introns_df["seqname"] == seq_id]
        for _, intron in seq_introns.iterrows():
            start_idx = intron["start"] - 1
            end_idx = intron["end"]
            #bound check
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(sequence):
                end_idx = len(sequence)
            labels[start_idx:end_idx] = 'I'
            
        labeled_sequences[seq_id] = ''.join(labels)
    return labeled_sequences



# Load and process the unlabeled data (the raw sequences as TTCATAATTA)
genome_sequences = parse_fasta(genome_fasta)

# Mapping of Roman numerals to integers to be used to separate training and testing data
roman_to_integer = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14,
    'XV': 15, 'XVI': 16
}

# Separate training and testing data using dictionary 

#training_data: chromosomes 1-12
training_data = {seq_id: seq for seq_id, seq in genome_sequences.items() 
                 if roman_to_integer.get(seq_id, 0) in range(1, 13)}

#testing data: chromosomes 13-16
testing_data = {seq_id: seq for seq_id, seq in genome_sequences.items() 
                if roman_to_integer.get(seq_id, 0) in range(13, 17)}

#parse the labeled data
gtf_data = parse_gtf(gtf_file)

#split the gtf_data into training and testing
training_gtf_data = gtf_data[gtf_data['seqname'].map(roman_to_integer.get).isin(range(1, 13))]
testing_gtf_data = gtf_data[gtf_data['seqname'].map(roman_to_integer.get).isin(range(13, 17)) ]                    

exons, introns = extract_exon_intron(training_gtf_data)
testing_exons, testing_introns = extract_exon_intron(testing_gtf_data)

print(introns.head())
# Display summaries
print("Genome Sequences:", list(genome_sequences.keys()))
# print("Exons:\n", exons)
# print("Introns:\n", introns)

#go through .gtf file and label introns and exons and intergenic regions
# Label the sequences
labeled_sequences = label_fasta_sequences(training_data, exons, introns)
labeled_testing_sequences = label_fasta_sequences(testing_data, testing_exons, testing_introns)

#encoded_string = ''.join([''.join(labels) for seq_id, labels in labeled_sequences.items()])


#for seq_id, labeled_sequence in labeled_sequences.items():
    #print(f"Sequence ID: {seq_id}")
    #print("Labels:")
    #print("".join(labeled_sequence))  # Convert list of labels back to a string for display
    #print("=" * 50)


#PART 2: define the components for the HMM (hidden states,)

states = ['E', 'I', 'U']
nucleotides = ['A', 'C', 'G', 'T']

#initialize emission counts and transition counts
emission_counts = {s: Counter() for s in states}
state_counts = Counter()
transition_counts = {s: Counter() for s in states}

for seq_id, seq in training_data.items():
    labels = labeled_sequences[seq_id]
    # Update emission counts - amount of times symbol (ATCG) is emmited from hidden state (I, E, U) 
    for i, (nuc, st) in enumerate(zip(seq, labels)):
        if st in states and nuc in nucleotides:
            emission_counts[st][nuc] += 1 #
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


#initial probabilities - tracking which states start each sequence.
initial_counts = Counter()
num_sequences = len(training_data)
for seq_id, seq in training_data.items():
     start_state = labeled_sequences[seq_id][0]
     initial_counts[start_state] += 1

initial_probs = np.zeros(len(states))
for i, s in enumerate(states):
    initial_probs[i] = initial_counts[s] / num_sequences if num_sequences > 0 else (1.0 / len(states))

#convert transition_probs dict to a numpy array for hmm
transition_mat = np.zeros((len(states), len(states)))
for i, s in enumerate(states):
    for j, s2 in enumerate(states):
        transition_mat[i, j] = transition_probs[s][s2]

#convert emission_probs dict to a numpy array
emission_mat = np.zeros((len(states), len(nucleotides)))
for i, s in enumerate(states):
    for j, nuc in enumerate(nucleotides):
        emission_mat[i, j] = emission_probs[s][nuc]

#validate each row sums to 1 to make sure probabilities are reasonable
print(emission_mat)
print(initial_probs)
print(transition_mat)

#create hmm model with given probabilities- finally yay!
model = hmm.CategoricalHMM(n_components=len(states), init_params="")
model.startprob_ = initial_probs
model.transmat_ = transition_mat
model.emissionprob_ = emission_mat


#map nucleotides to integers so we can test the model 
mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
#encode the testing_data 
encoded_sequence = np.array([[mapping[nuc]] for seq in testing_data.values() for nuc in seq], dtype=int)


#predict the hidden state path using the model
hidden_states = model.predict(encoded_sequence)
mapping = {0: 'E', 1: 'I', 2: 'U'}
decoded_states = [mapping[state] for state in hidden_states]

decoded_string = ''.join(decoded_states)

true_labels = [label for seq_id in labeled_testing_sequences for label in labeled_testing_sequences[seq_id]]
pred_labels = decoded_states  # ensure this is a list of same length, ['E', 'I', 'I', 'E', 'U', 'U']

acc = accuracy_score(true_labels, pred_labels)
print("Accuracy:", acc)
