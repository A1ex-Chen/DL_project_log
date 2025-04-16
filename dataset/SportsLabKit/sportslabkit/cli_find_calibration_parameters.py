def find_calibration_parameters(self, checkerboard_files: str, output: str,
    fps: int=1, scale: int=1, pts: int=50, calibration_method: str='zhang'):
    """_summary_

        Args:
            checkerboard_files (str): Path to the checkerboard video (wildcards are supported).
            output (str): Path to save the calibration parameters.
            fps (int, optional): _description_. Defaults to 1.
            scale (int, optional): _description_. Defaults to 1.
            pts (int, optional): _description_. Defaults to 50.
            calibration_method (str, optional): _description_. Defaults to "zhang".
        """
    mtx, dist, mapx, mapy = find_intrinsic_camera_parameters(checkerboard_files
        , fps=fps, scale=scale, save_path=False, draw_on_save=False,
        points_to_use=pts, calibration_method=calibration_method,
        return_mappings=True)
    dirname = os.path.dirname(output)
    if len(dirname) != 0:
        os.makedirs(dirname, exist_ok=True)
    np.savez(output, mtx=mtx, dist=dist, mapx=mapx, mapy=mapy)
