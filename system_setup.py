import numpy as np
import math

def rotate_point(x, y, center_x, center_y, alpha_radians):
    new_x = (x - center_x) * math.cos(alpha_radians) - (y - center_y) * math.sin(alpha_radians) + center_x
    new_y = (x - center_x) * math.sin(alpha_radians) + (y - center_y) * math.cos(alpha_radians) + center_y

    return new_x, new_y

class pdb_atom:
    def __init__(self, linetype, atom_num, atom_type, res_type, mol_num, res_num, x, y, z, skip1, skip2, chain_num, element):
        self.atom_num = atom_num        # Float, atom number
        self.atom_type = atom_type      # String, atom type (not element, uses topology naming)
        self.res_num = res_num          # Float, residue number
        self.res_type = res_type        # String, residue type (uses topology naming)
        self.x = x                      # Float
        self.y = y                      # Float
        self.z = z                      # Float
        self.linetype = linetype        # string, will be 'ATOM' for most cases (could be marking ions)
        self.skip1 = skip1              # Float, data entry in pdb that we don'r use
        self.skip2 = skip2              # Float, data entry in pdb that we don't use 
        self.chain_num = chain_num      # string, needs to be converted, chains are counted from 0
        self.element = element          # string
        self.mol_num = mol_num          # string, molecule ID, important when there are diff types of cellulose in the system
        
    def rotate(self,plane,center=(0.,0.),alpha=0.):
        #alpha in radians, center is a tuple of two(!) coordinates: x,y x,z or y,z, "plane" is the perpendicular axis
        if plane == "z":
            self.x, self.y = rotate_point(self.x,self.y,center[0],center[1],alpha)
        elif plane == "y":
            self.x, self.z = rotate_point(self.x,self.z,center[0],center[1],alpha)
        elif plane == "x":
            self.y, self.z = rotate_point(self.y,self.z,center[0],center[1],alpha)
            
    def coordinate_shift(self,x_shift=0.,y_shift=0.,z_shift=0.):
        self.x = self.x + x_shift
        self.y = self.y + y_shift
        self.z = self.z + z_shift
        
    def number_shift(self,shift=0.):
        self.atom_num = self.atom_num + shift
        
        
def parse_textfile_line(line):
    data = line.split()
    parsed_data = []

    for item in data:
        try:
            # Try to convert the item to a float
            parsed_item = float(item)
        except ValueError:
            # If it's not a float, keep it as a string
            parsed_item = item

        # Append the parsed item to the result list
        parsed_data.append(parsed_item)

    return parsed_data

def read_pdb(filename):
    atoms = []
    with open(filename, 'r') as file:
        for line in file:
            data_line = parse_textfile_line(line)
            if len(data_line) == 12: #when the res_type and the mol_num columns collide this splits them
                temp = data_line[3][:-1]
                data_line[3] = data_line[3][-1:]
                data_line.insert(3,temp)
            if data_line[0] != 'ATOM':
                continue
            else:
                atom = pdb_atom(*data_line)
                atoms.append(atom)
    return atoms

def write_pdb(atoms,output_file):
    with open(output_file, "w") as file:
        for atom in atoms:
            res_num_string = str(int(atom.res_num))
            if len(res_num_string) == 1:
                res_num_string = " "+res_num_string
                    
            file.write(f"{atom.linetype}{str(int(atom.atom_num)):>7}  {atom.atom_type:<3} {atom.res_type} {atom.mol_num}{str(int(atom.res_num)):>4}    {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}  {atom.skip1:3.2f}  {atom.skip2:3.2f}      {atom.chain_num}   {atom.element}\n")
        file.write(f"END")     
    print(".pdb file written :)")

def only_selected_chains(atoms,list_of_chains):
    selected_atoms = []
    for item in atoms:
        if item.chain_num in list_of_chains:
            selected_atoms.append(item)
    return selected_atoms

def generate_index_file(atoms,list_of_chains,group_name,output_filename):
    selected_atoms = []
    for item in atoms:
        if item.chain_num in list_of_chains:
            selected_atoms.append(str(int(atom.atom_num)))
    with open(output_filename,'w') as file:
        file.write('[ '+group_name+' ]\n\n')
        for item in selected:
            file.write(item + ' ')
    file.close()