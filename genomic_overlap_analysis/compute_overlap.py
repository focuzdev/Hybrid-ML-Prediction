"""
Genomic Overlap Analysis for Predicted smORFs
=============================================

This script takes the 557 predicted smORFs (from ORFipy) and checks their 
coordinates against the M. tuberculosis H37Rv reference annotation (mycobact.gff).

Outputs:
1. smorf_gene_overlap_results.csv  (Tabular summary of all 557 smORFs)
2. overlapping_smORFs.fasta        (FASTA sequences that overlap known genes)
3. unannotated_smORFs.fasta        (FASTA sequences with no overlapping genes)
"""

import re
import pandas as pd
import os
from datetime import datetime

def main():
    # File paths
    gff_file = "mycobact.gff"
    fasta_file = "mtb_predicted_smORFs.fasta"
    
    if not os.path.exists(gff_file) or not os.path.exists(fasta_file):
        print(f"Error: Make sure {gff_file} and {fasta_file} are in the current directory.")
        return

    # 1. Load reference GFF
    print("Loading reference GFF...")
    genes = pd.read_csv(gff_file, sep="\t", comment="#", 
                        names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"])
    # Keep only annotated 'gene' features
    genes = genes[genes["type"] == "gene"]
    print(f"Loaded {len(genes)} annotated genes.")

    # 2. Parse FASTA and check overlaps
    print("\nParsing FASTA headers...")
    # Regex expects headers like: >mtb_smORF3_33964_34218_+
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
                # Process the previous record before starting the new one
                if current_header:
                    # Check overlap
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

                # Start new record
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

        # Process the very last record in the file
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

    # 3. Calculate statistics
    X = len(overlapping_records)
    Z = len(unannotated_records)
    total = X + Z
    Y = (X / total) * 100 if total > 0 else 0
    W = (Z / total) * 100 if total > 0 else 0

    print("\n" + "="*40)
    print("=== OVERLAP ANALYSIS RESULTS ===")
    print("="*40)
    print(f"Total smORFs parsed:          {total}")
    print(f"Overlapping annotated genes:  {X} ({Y:.1f}%)")
    print(f"Unannotated (no gene overlap): {Z} ({W:.1f}%)")
    print("="*40 + "\n")

    # 4. Save to CSV (Comma-separated for Excel)
    csv_filename = "smorf_gene_overlap_results.csv"
    df_results = pd.DataFrame(csv_results)
    
    try:
        df_results.to_csv(csv_filename, sep=",", index=False)
        print(f"Saved: {csv_filename}")
    except PermissionError:
        timestamp = datetime.now().strftime("%H%M%S")
        csv_filename = f"smorf_gene_overlap_results_{timestamp}.csv"
        df_results.to_csv(csv_filename, sep=",", index=False)
        print(f"Original file open in Excel. Saved as: {csv_filename}")

    # 5. Save FASTA files
    with open("overlapping_smORFs.fasta", "w") as f:
        for h, s in overlapping_records:
            f.write(f"{h}\n{s}\n")
    print("Saved: overlapping_smORFs.fasta")

    with open("unannotated_smORFs.fasta", "w") as f:
        for h, s in unannotated_records:
            f.write(f"{h}\n{s}\n")
    print("Saved: unannotated_smORFs.fasta")

if __name__ == "__main__":
    main()