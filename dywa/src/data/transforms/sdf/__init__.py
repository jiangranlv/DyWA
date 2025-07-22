# [1] Trimsh = slow. But most reliable.
# from .sdf.sdf_naive_trimesh import SignedDistanceTransform
# [2] PySDF = mysterious Qhull error?
# from .sdf_pysdf import SignedDistanceTransform
# [3] O3D = issue with multiprocessing I think.
# O3D --> for some reason, too many false positives (considered "inside")
# from .sdf_o3d import SignedDistanceTransform
