class Udf_Query:
    def __init__(self, ):
        pass

    @staticmethod
    def query(
        mesh_vertices,
        mesh_faces,
        query_points,
    ):
        '''
        Query the signed distance from query points to the mesh surface.
        '''
        import meshsdf_loss_cuda  # lazy import: only needed for CUDA SDF path (check_self_penetration_igl)
        outputs = meshsdf_loss_cuda.forward(
            mesh_vertices,
            mesh_faces,
            query_points,
        )
        dist, asso = outputs[1], outputs[2]

        return dist, asso
