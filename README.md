# HMM with Viterbi Algorithm

This repository contains our implementation of a Hidden Markov Model (HMM) utilizing the Viterbi algorithm. The code processes DNA sequences and annotations from the yeast genome (*Saccharomyces cerevisiae*).  

## Files  

### Code  
- `bioinformatics_viterbi.py`  
  The final implementation of the HMM model integrated with the Viterbi algorithm.  
- `bioinformatics.py` 
  The initial version of the HMM model without the Viterbi algorithm.  

### Data  
- `Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa`  
  Contains unannotated DNA sequences of the yeast (*Saccharomyces cerevisiae*) genome.  
- `Saccharomyces_cerevisiae.R64-1-1.113.gtf`  
  Includes gene annotations for the yeast genome.  

### Results  
- `final_results.png`  
  Visual representation of the final outcomes from our implementation.  

## Usage  
1. Run `bioinformatics_viterbi.py` to apply the HMM with the Viterbi algorithm on the provided DNA sequence and annotation data.  
2. Compare results with `final_results.png` to validate predictions.  

## Summary  
This project demonstrates the application of HMMs and the Viterbi algorithm to gene prediction tasks in bioinformatics. It serves as a foundation for further research and development in computational genomics.  
