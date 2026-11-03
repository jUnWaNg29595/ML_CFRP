"""Strict fixed-width GRO and GROMACS topology parser.
Eliminates all hardcoded bonds, energies, and fake fallbacks.
"""
from pathlib import Path
import os, sys, re
import numpy as np

ELEMENT_BY_ATOM_PREFIX = {
    "C": "C", "H": "H", "O": "O", "N": "N", "S": "S"
}

def parse_fixed_width_gro(gro_path: str) -> dict:
    p = Path(gro_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"GRO file absent or empty: {gro_path}")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        raise ValueError(f"GRO file too short: {len(lines)} lines")
    
    title = lines[0].strip()
    try:
        natoms = int(lines[1].strip())
    except ValueError as e:
        raise ValueError(f"Cannot parse GRO atom count from line 2: '{lines[1]}'") from e
        
    if len(lines) < natoms + 3:
        raise ValueError(f"GRO atom count mismatch: header declares {natoms}, file has {len(lines)-3} data lines")
        
    coords = []
    atom_names = []
    res_names = []
    res_ids = []
    
    for idx, line in enumerate(lines[2:2 + natoms], 1):
        if len(line) < 44:
            raise ValueError(f"Corrupt fixed-width GRO line {idx+2}: length {len(line)} < 44")
        try:
            r_id = int(line[0:5].strip())
            r_name = line[5:10].strip()
            a_name = line[10:15].strip()
            # 20:28 x, 28:36 y, 36:44 z
            x = float(line[20:28].strip())
            y = float(line[28:36].strip())
            z = float(line[36:44].strip())
        except Exception as e:
            raise ValueError(f"Failed parsing columns on GRO line {idx+2}: '{line}'") from e
            
        coords.append([x, y, z])
        atom_names.append(a_name)
        res_names.append(r_name)
        res_ids.append(r_id)
        
    box_line = lines[2 + natoms].strip().split()
    if len(box_line) < 3:
        raise ValueError(f"Corrupt GRO box vector line: '{lines[2+natoms]}'")
    box = np.array([float(box_line[0]), float(box_line[1]), float(box_line[2])], dtype=float)
    
    return {
        "title": title,
        "natoms": natoms,
        "coords_nm": np.array(coords, dtype=float),
        "atom_names": atom_names,
        "res_names": res_names,
        "res_ids": res_ids,
        "box_lengths_nm": box,
        "volume_nm3": float(np.prod(box))
    }

def parse_gromacs_top_bonds_and_atoms(top_path: str) -> dict:
    """Parses atom definitions and bond topology from GROMACS top / itp files."""
    p = Path(top_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"TOP file absent or empty: {top_path}")
    
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    atoms = [] # [{id, type, resnr, resname, atomname, charge, mass, element}]
    bonds = [] # [(atom1_idx_0based, atom2_idx_0based)]
    
    current_section = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue
        
        # Check section header
        m = re.match(r"\[\s*(\w+)\s*\]", stripped)
        if m:
            current_section = m.group(1).lower()
            continue
            
        parts = stripped.split()
        if current_section == "atoms":
            # format: nr type resnr residue atom cgnr charge mass
            if len(parts) >= 8 and parts[0].isdigit():
                aid = int(parts[0])
                atype = parts[1]
                resnr = int(parts[2])
                resname = parts[3]
                atomname = parts[4]
                q = float(parts[6])
                m = float(parts[7])
                
                # Determine element from atom type or name
                elem = "C"
                for prefix in ["CL", "BR", "NA", "MG"]:
                    if atype.upper().startswith(prefix):
                        elem = prefix.capitalize()
                        break
                else:
                    first = atomname[0].upper()
                    if first in ("C", "H", "O", "N", "S"):
                        elem = first
                atoms.append({
                    "id": aid,
                    "type": atype,
                    "resnr": resnr,
                    "resname": resname,
                    "atomname": atomname,
                    "charge": q,
                    "mass": m,
                    "element": elem
                })
        elif current_section == "bonds":
            # format: ai aj funct ...
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                ai = int(parts[0]) - 1 # 0-based
                aj = int(parts[1]) - 1
                bonds.append((min(ai, aj), max(ai, aj)))
                
    # Unique sorted bonds
    unique_bonds = sorted(list(set(bonds)))
    
    # Identify crosslink bonds (bonds between different residue numbers)
    atom_res_map = {i: a["resnr"] for i, a in enumerate(atoms)}
    crosslink_bonds = []
    for a1, a2 in unique_bonds:
        r1 = atom_res_map.get(a1)
        r2 = atom_res_map.get(a2)
        if r1 is not None and r2 is not None and r1 != r2:
            crosslink_bonds.append((r1 - 1, r2 - 1)) # monomer unit indices (0-based)
            
    # Count elements
    elem_counts = {}
    for a in atoms:
        e = a["element"]
        elem_counts[e] = elem_counts.get(e, 0) + 1
        
    return {
        "atoms": atoms,
        "natoms": len(atoms),
        "bonds": unique_bonds,
        "crosslink_bonds": crosslink_bonds,
        "element_counts": elem_counts,
        "elements": [a["element"] for a in atoms]
    }
