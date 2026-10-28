import re
import pandas as pd
import os
from datetime import datetime

"""
Genomic Overlap Analysis for Predicted smORFs
=============================================

This script evaluates the 557 smORFs predicted to determine how many overlap with 
annotated protein-coding genes in the M. tuberculosis H37Rv reference genome.

Input files:
  - mtb_predicted_smORFs.fasta : Contains the 557 smORF sequences and coordinates.
  - mycobact.gff               : The NCBI reference annotation for NC_000962.3.

Output files:
  - smorf_gene_overlap_results.csv : Table of all 557 smORFs and their overlap status.
  - overlapping_smORFs.fasta       : Subset of sequences overlapping annotated genes.
  - unannotated_smORFs.fasta       : Subset of sequences with no gene overlap.
"""

def run_overlap_analysis(gff_file="mycobact.gff", fasta_file="mtb_predicted_smORFs.fasta"):
    # Check for required inputs
    if not os.path.exists(gff_file) or not os.path.exists(fasta_file):
        raise FileNotFoundError(f"Missing input files. Ensure {gff_file} and {fasta_file} are in the directory.")

    print("Loading reference GFF...")
    genes = pd.read_csv(gff_file, sep="\t", comment="#", 
                        names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"])
    
    # Filter for annotated 'gene' features only to avoid double-counting exons/CDS
    genes = genes[genes["type"] == "gene"]
    print(f"Loaded {len(genes)} annotated genes.")

    print("Parsing FASTA and checking overlaps...")
    # Regex matches headers formatted as >mtb_smORF1_10966_11121_+
    pattern = re.compile(r">(mtb_smORF\d+)_(\d+)_(\d+)_([+-])") 

    csv_results = []
    overlapping_records = []
    unannotated_records = []
    
    current_header = None
    current_seq = []
    prev_id = prev_start = prev_end = prev_strand = None

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Process the previous sequence before starting the new header
                if current_header:
                    overlaps = (genes["start"] <= prev_end) & (genes["end"] >= prev_start)
                    status = "Overlapping" if overlaps.any() else "Non-overlapping"
                    
                    csv_results.append({
                        "smORF_ID": prev_id,
                        "Start": prev_start,
                        "End": prev_end,
                        "Strand": prev_strand,
                        "Overlap_status": status
                    })
                    
                    if status == "Overlapping":
                        overlapping_records.append((current_header, "".join(current_seq)))
                    else:
                        unannotated_records.append((current_header, "".join(current_seq)))

                # Start parsing the new header
                current_header = line
                current_seq = []
                
                match = pattern.search(line)
                if match:
                    prev_id = match.group(1)
                    prev_start = int(match.group(2))
                    prev_end = int(match.group(3))
                    prev_strand = match.group(4)
            else:
                current_seq.append(line)

        # Process the final sequence in the file
        if current_header and current_seq:
            overlaps = (genes["start"] <= prev_end) & (genes["end"] >= prev_start)
            status = "Overlapping" if overlaps.any() else "Non-overlapping"
            
            csv_results.append({
                "smORF_ID": prev_id,
                "Start": prev_start,
                "End": prev_end,
                "Strand": prev_strand,
                "Overlap_status": status
            })
            
            if status == "Overlapping":
                overlapping_records.append((current_header, "".join(current_seq)))
            else:
                unannotated_records.append((current_header, "".join(current_seq)))

    # Calculate summary statistics
    X = len(overlapping_records)
    Z = len(unannotated_records)
    total = X + Z
    Y = (X / total) * 100 if total > 0 else 0
    W = (Z / total) * 100 if total > 0 else 0

    print("\n" + "="*40)
    print("OVERLAP ANALYSIS RESULTS")
    print("="*40)
    print(f"Total smORFs:                 {total}")
    print(f"Overlapping annotated genes:  {X} ({Y:.1f}%)")
    print(f"Unannotated (no gene overlap): {Z} ({W:.1f}%)")
    print("="*40 + "\n")

    # Save results to CSV (comma-separated for Excel compatibility)
    csv_filename = "smorf_gene_overlap_results.csv"
    df_results = pd.DataFrame(csv_results)

    try:
        df_results.to_csv(csv_filename, sep=",", index=False)
        print(f"Saved: {csv_filename}")
    except PermissionError:
        # Fallback if the file is currently open in Excel
        timestamp = datetime.now().strftime("%H%M%S")
        csv_filename = f"smorf_gene_overlap_results_{timestamp}.csv"
        df_results.to_csv(csv_filename, sep=",", index=False)
        print(f"File open in Excel. Saved as: {csv_filename}")

    # Save the separated FASTA files
    with open("overlapping_smORFs.fasta", "w") as f:
        for h, s in overlapping_records:
            f.write(f"{h}\n{s}\n")
    print("Saved: overlapping_smORFs.fasta")

    with open("unannotated_smORFs.fasta", "w") as f:
        for h, s in unannotated_records:
            f.write(f"{h}\n{s}\n")
    print("Saved: unannotated_smORFs.fasta")

if __name__ == "__main__":
    run_overlap_analysis()
