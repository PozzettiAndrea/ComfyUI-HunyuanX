# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy
import trimesh
    
def postprocessmesh(vertices: np.array, faces: np.array, settings):
    print('Generating Meshlib Mesh ...')
    mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
    print('Packing Optimally ...')
    mesh.packOptimally()
    print('Decimating ...')
    mrmeshpy.decimateMesh(mesh, settings)
    
    out_verts = mrmeshnumpy.getNumpyVerts(mesh)
    out_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)
    
    mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces)   
    print(f"Reduced faces, resulting in {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces")
        
    return mesh


