// This code conforms with the UFC specification version 2018.2.0.dev0
// and was automatically generated by FFCx version 0.1.1.dev0.
//
// This code was generated with the following parameters:
//
//  {'assume_aligned': -1,
//   'epsilon': 1e-14,
//   'output_directory': '.',
//   'padlen': 1,
//   'profile': False,
//   'scalar_type': 'double',
//   'table_atol': 1e-09,
//   'table_rtol': 1e-06,
//   'tabulate_tensor_void': False,
//   'ufl_file': ['SaintVenantKirchhoff_Tetra_Order2.ufl'],
//   'verbosity': 30,
//   'visualise': False}


#pragma once

typedef double ufc_scalar_t;
#include <ufc.h>

#ifdef __cplusplus
extern "C" {
#endif

extern ufc_finite_element element_680feec8a499f5067d6dcbd42a688e3a74aed060;

extern ufc_finite_element element_769533c7766d1e3908ac031c78a90a219ef91f1c;

extern ufc_finite_element element_637854f3d0707fd1767e3602c81b0e1c20f574f6;

extern ufc_finite_element element_7028612f97b89a71fc2f49d756806b9bbd6e45c5;

extern ufc_dofmap dofmap_680feec8a499f5067d6dcbd42a688e3a74aed060;

extern ufc_dofmap dofmap_769533c7766d1e3908ac031c78a90a219ef91f1c;

extern ufc_dofmap dofmap_637854f3d0707fd1767e3602c81b0e1c20f574f6;

extern ufc_dofmap dofmap_7028612f97b89a71fc2f49d756806b9bbd6e45c5;

extern ufc_integral integral_d1a10dd84882beee04265057c36795fc66ace1f3;

extern ufc_integral integral_e0fe1851a68659299a868c0c08b301dfee8bb209;

extern ufc_form form_be74d71ef59d35685265ccce914fe85a5a5a8f22;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufc_form* form_SaintVenantKirchhoff_Tetra_Order2_F;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufc_function_space* functionspace_form_SaintVenantKirchhoff_Tetra_Order2_F(const char* function_name);

extern ufc_form form_ec99f6bacf04a4191ddcd17b268f3b2573e30ae2;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufc_form* form_SaintVenantKirchhoff_Tetra_Order2_J;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufc_function_space* functionspace_form_SaintVenantKirchhoff_Tetra_Order2_J(const char* function_name);

#ifdef __cplusplus
}
#endif
