import os
import sys
import numpy as np
from copy import deepcopy
from simnibs import mesh_io, run_simnibs, sim_struct
from simnibs.utils import TI_utils as TI

###########################################

# Simplified TI Simulation Script for Testing
# This script runs a single SimNIBS simulation with hardcoded parameters

###########################################

# HARDCODED PATHS - MODIFY THESE AS NEEDED
subject_id = "101"  # Your subject ID
project_dir = "/Users/idohaber/BIDS_new"  # Your project directory
simulation_dir = "/Users/idohaber/BIDS_new/derivatives/SimNIBS/sub-101/TI_test_output"  # Output directory

# Simulation parameters
sim_type = "scalar"  # Options: 'scalar', 'vn', 'dir', 'mc'
intensity1 = 0.008  # Current intensity for first pair (8mA = 0.008A)
intensity2 = 0.008  # Current intensity for second pair (8mA = 0.008A)
electrode_shape = "ellipse"  # Options: 'rect' or 'ellipse'
dimensions = [20, 20]  # Electrode dimensions [x, y] in mm
thickness = 2.0  # Electrode thickness in mm
eeg_net = "GSN-HydroCel-256.csv"  # EEG net filename

# Montage definition - YOUR SPECIFIC MONTAGE
# Format: [[electrode1_pair1, electrode2_pair1], [electrode1_pair2, electrode2_pair2]]
# Your montage: E070-E095, E034-E020
montage = [["E070", "E095"], ["E034", "E020"]]  # Your electrode pairs

# Base paths
derivatives_dir = os.path.join(project_dir, 'derivatives')
simnibs_dir = os.path.join(derivatives_dir, 'SimNIBS', f'sub-{subject_id}')
base_subpath = os.path.join(simnibs_dir, f'm2m_{subject_id}')
conductivity_path = base_subpath
tensor_file = os.path.join(conductivity_path, "DTI_coregT1_tensor.nii.gz")

# Create output directory
os.makedirs(simulation_dir, exist_ok=True)

def run_simulation():
    print(f"Starting simulation for subject: {subject_id}")
    
    S = sim_struct.SESSION()
    S.subpath = base_subpath
    S.anisotropy_type = sim_type
    
    print(f"Set up SimNIBS session with anisotropy type: {sim_type}")
    
    # Use simulation directory for output
    S.pathfem = simulation_dir
    
    # Use the selected EEG net
    S.eeg_cap = os.path.join(base_subpath, "eeg_positions", eeg_net)
    
    S.map_to_surf = True
    S.map_to_fsavg = False
    S.map_to_vol = True
    S.map_to_mni = True
    S.open_in_gmsh = False
    S.tissues_in_niftis = "all"

    # Load the conductivity tensors
    S.dti_nii = tensor_file

    # First electrode pair
    tdcs = S.add_tdcslist()
    tdcs.anisotropy_type = sim_type
    
    # Set currents for first pair
    tdcs.currents = [intensity1, -intensity1]
    
    # First electrode
    electrode = tdcs.add_electrode()
    electrode.channelnr = 1
    electrode.centre = montage[0][0]
    electrode.shape = electrode_shape
    electrode.dimensions = dimensions
    electrode.thickness = [thickness, 2]

    # Second electrode
    electrode = tdcs.add_electrode()
    electrode.channelnr = 2
    electrode.centre = montage[0][1]
    electrode.shape = electrode_shape
    electrode.dimensions = dimensions
    electrode.thickness = [thickness, 2]

    # Second electrode pair
    tdcs_2 = S.add_tdcslist(deepcopy(tdcs))
    # Set currents for second pair
    tdcs_2.currents = [intensity2, -intensity2]
    tdcs_2.electrode[0].centre = montage[1][0]
    tdcs_2.electrode[1].centre = montage[1][1]

    print("Running SimNIBS simulation...")
    run_simnibs(S)
    print("SimNIBS simulation completed")

    subject_identifier = base_subpath.split('_')[-1]
    anisotropy_type = S.anisotropy_type

    m1_file = os.path.join(S.pathfem, f"{subject_identifier}_TDCS_1_{anisotropy_type}.msh")
    m2_file = os.path.join(S.pathfem, f"{subject_identifier}_TDCS_2_{anisotropy_type}.msh")

    print("Loading mesh files for TI calculation")
    m1 = mesh_io.read_msh(m1_file)
    m2 = mesh_io.read_msh(m2_file)

    # Keep relevant tissue tags
    tags_keep = np.hstack((np.arange(1, 100), np.arange(1001, 1100)))
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    ef1 = m1.field["E"]
    ef2 = m2.field["E"]
    
    print("Calculating TI vectors using get_dirTI")
    
    # Calculate TI vectors using get_dirTI
    # We'll use the original E field directions as the direction vectors
    # This will give us the TI magnitude in the direction of the original E fields
    TI_vectors = TI.get_dirTI(ef1.value, ef2.value, ef1.value)
    
    print(f"Calculated TI vectors for {len(TI_vectors)} elements")
    print(f"TI_vector range: {np.min(TI_vectors):.6f} to {np.max(TI_vectors):.6f} V/m")

    # Write output mesh file with TI_vector field
    print("Writing output mesh file")
    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TI_vectors, "TI_vector")
    
    output_file = os.path.join(S.pathfem, "TI_vector.msh")
    mesh_io.write_msh(mout, output_file)

    # Create visualization
    v = mout.view(visible_tags=[1002, 1006], visible_fields=["TI_vector"])
    v.write_opt(output_file)
    
    print(f"Completed simulation")
    print(f"Output saved to: {output_file}")
    print(f"Mesh contains TI_vector field")

if __name__ == "__main__":
    # Check if required files exist
    if not os.path.exists(base_subpath):
        print(f"Error: Base subject path not found: {base_subpath}")
        print("Please check your project_dir and subject_id settings")
        sys.exit(1)
    
    if not os.path.exists(tensor_file):
        print(f"Error: Tensor file not found: {tensor_file}")
        print("Please ensure DTI_coregT1_tensor.nii.gz exists in the subject directory")
        sys.exit(1)
    
    eeg_cap_file = os.path.join(base_subpath, "eeg_positions", eeg_net)
    if not os.path.exists(eeg_cap_file):
        print(f"Error: EEG cap file not found: {eeg_cap_file}")
        print("Please check your eeg_net setting")
        sys.exit(1)
    
    # Run the simulation
    try:
        run_simulation()
        print("Simulation completed successfully!")
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.exit(1) 