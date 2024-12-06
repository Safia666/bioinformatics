from Bio import SeqIO
import pandas as pd
import re

#contains the nucleotide sequences of the yeast genome - FASTA file
fasta_file = "/Users/safiam/Desktop/bioinformatics /Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"

#contains annotations of genes, exons, introns, etc of the yeast genome - GTF file
gtf_file = "/Users/safiam/Desktop/bioinformatics /Saccharomyces_cerevisiae.R64-1-1.113.gtf"

#parse the FASTA file
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
    #filter exons through feature
    exons = gtf_df[gtf_df["feature"] == "exon"].copy()

    #sort exons by location for processing introns
    exons.sort_values(by=["seqname", "strand", "start"], inplace=True)

    introns = []
    #group exons by seqname, strand, and transcript_id to find introns
    for (seqname, strand, transcript_id), group in exons.groupby(["seqname", "strand", "transcript_id"]):
        sorted_exons = group.sort_values(by="start") #sort the exons in each group
        for i in range(len(sorted_exons) - 1): #find all the gaps between exons to be introns
            intron_start = int(sorted_exons.iloc[i]["end"]) + 1
            intron_end = int(sorted_exons.iloc[i + 1]["start"]) - 1
            introns.append({ #add each intron 
                "seqname": seqname,
                "source": "inferred",
                "feature": "intron",
                "start": intron_start,
                "end": intron_end,
                "strand": strand,
                "transcript_id": transcript_id
            })

    introns_df = pd.DataFrame(introns) #make datafram with introns
    return exons, introns_df
   


# Load and process the data
genome_sequences = parse_fasta(fasta_file)
gtf_data = parse_gtf(gtf_file)
exons, introns = extract_exon_intron(gtf_data)
exons.sort_values(by=["seqname", "transcript_id", "start"], inplace=True)
introns.sort_values(by=["seqname", "transcript_id", "start"], inplace=True)

# Display summaries
print("Genome Sequences:", list(genome_sequences.keys()))
print("Exons:\n", exons.head())
print("Introns:\n", introns.head())
