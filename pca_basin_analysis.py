#!/usr/bin/env python
"""
PCA Basin Analysis for MD Trajectories
Original script for interactive basin selection and free energy calculation.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import subprocess as sp
import argparse
import sys

# Constants
kT = 2.479  # 300 K in kJ/mol

# Parse command line arguments
parser = argparse.ArgumentParser(description='PCA Basin Analysis')
parser.add_argument('--trj', required=True, help='Combined trajectory (.xtc)')
parser.add_argument('--tpr', required=True, help='Topology file (.tpr)')
parser.add_argument('--ndx', default=None, help='Index file (optional)')
parser.add_argument('--grp', type=int, default=1, help='trjconv output group number')
parser.add_argument('--apo_file', default='apo.dat', help='Apo PCA data file')
parser.add_argument('--holo_file', default='holo.dat', help='Holo PCA data file')
parser.add_argument('--pca_all_file', default='pc12_all.txt', help='All frames PCA data')
args = parser.parse_args()

# ---------- Load PCA data ----------
print("Loading PCA data...")
try:
    apo = np.loadtxt(args.apo_file, usecols=(0, 1))
    holo = np.loadtxt(args.holo_file, usecols=(0, 1))
    all_data = np.loadtxt(args.pca_all_file)
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    sys.exit(1)

print(f"Loaded {len(apo)} apo and {len(holo)} holo conformations")

# ---------- Create histograms ----------
bins = 200
H_apo, xedg, yedg = np.histogram2d(apo[:, 0], apo[:, 1], bins=bins, density=False)
H_holo, _, _ = np.histogram2d(holo[:, 0], holo[:, 1], bins=bins, density=False)

xc = 0.5 * (xedg[1:] + xedg[:-1])
yc = 0.5 * (yedg[1:] + yedg[:-1])
XX, YY = np.meshgrid(xc, yc, indexing='ij')

# ---------- Function to extract PDBs ----------
def make_pdbs(name, frames):
    """Extract PDB frames using GROMACS trjconv."""
    # Create index file
    ndx_file = f'{name}.ndx'
    with open(ndx_file, 'w') as f:
        f.write(f'[ {name} ]\n')
        for i, frm in enumerate(frames, 1):
            f.write(f'{frm:6d}')
            if i % 15 == 0:
                f.write('\n')
        f.write('\n')
    
    # Run trjconv
    cmd = f'echo {args.grp} | gmx trjconv -s {args.tpr} -f {args.trj} ' \
          f'-o {name}_centered.pdb -n {ndx_file} -pbc mol'
    
    print(f'  Running: {cmd}')
    try:
        sp.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f'  -> {name}_centered.pdb ({len(frames)} frames)')
    except sp.CalledProcessError as e:
        print(f'  Error: {e.stderr}')

# ---------- Interactive basin selection ----------
def pick_and_pdb(hist, label):
    """Interactive basin selection and PDB extraction."""
    # Switch to interactive backend
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    basin_counts = []
    
    while True:
        plt.figure(figsize=(10, 8))
        plt.imshow(hist.T, origin='lower', aspect='auto',
                  extent=[xedg[0], xedg[-1], yedg[0], yedg[-1]], 
                  cmap='Blues')
        plt.title(f'Select {label} basin - Left: add, Right: remove, Middle/Enter: close')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(label='Counts')
        
        # Get polygon points
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True, 
                        mouse_add=1, mouse_pop=3)
        
        if len(pts) < 3:
            print("Need at least 3 points. Skipping.")
            plt.close()
            break
        
        # Create polygon and find points inside
        poly = Path(pts)
        points = np.column_stack([XX.ravel(), YY.ravel()])
        inside = poly.contains_points(points).reshape(hist.shape)
        
        counts = hist[inside].sum()
        print(f'  Raw counts: {counts}')
        
        # Find frames inside polygon
        inside_all = poly.contains_points(all_data)
        frames = np.where(inside_all)[0] + 1  # 1-based frame numbers
        print(f'  Contains {len(frames)} frames')
        
        # Ask to extract PDBs
        ans = input('Extract PDBs? (y/n): ').strip().lower()
        if ans == 'y' and len(frames) > 0:
            basename = f'{label}_basin{len(basin_counts)+1}'
            make_pdbs(basename, frames)
        
        basin_counts.append(counts)
        plt.close()
        
        # Continue?
        cont = input(f'Continue with another {label} basin? (y/n): ').strip().lower()
        if cont != 'y':
            break
    
    return basin_counts

# ---------- Free energy calculation ----------
def calculate_free_energies(counts):
    """Calculate relative free energies from basin populations."""
    counts = np.array(counts, dtype=float)
    if len(counts) == 0:
        return []
    
    p = counts / counts.sum()
    # Avoid log(0)
    p = np.maximum(p, 1e-10)
    dg = -kT * np.log(p / p.max())
    
    return dg

# ---------- Main workflow ----------
print("\n" + "="*50)
print("Interactive PCA Basin Analysis")
print("="*50)

# Process apo basins
print("\n[APO State]")
apo_counts = []
if input("Analyze apo state? (y/n): ").strip().lower() == 'y':
    apo_counts = pick_and_pdb(H_apo, 'apo')

# Process holo basins
print("\n[HOLO State]")
holo_counts = []
if input("Analyze holo state? (y/n): ").strip().lower() == 'y':
    holo_counts = pick_and_pdb(H_holo, 'holo')

# Calculate free energies
print("\n" + "="*50)
print("Free Energy Results (relative to most stable basin)")
print("="*50)

if apo_counts:
    apo_dg = calculate_free_energies(apo_counts)
    print("\nAPO State:")
    for i, (count, dg_val) in enumerate(zip(apo_counts, apo_dg)):
        print(f'  Basin {i+1}: ΔG = {dg_val:6.2f} kJ/mol ({count} frames)')

if holo_counts:
    holo_dg = calculate_free_energies(holo_counts)
    print("\nHOLO State:")
    for i, (count, dg_val) in enumerate(zip(holo_counts, holo_dg)):
        print(f'  Basin {i+1}: ΔG = {dg_val:6.2f} kJ/mol ({count} frames)')

print("\nAnalysis complete!")
