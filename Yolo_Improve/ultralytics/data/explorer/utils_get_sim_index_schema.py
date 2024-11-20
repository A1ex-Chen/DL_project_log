def get_sim_index_schema():
    """Returns a LanceModel schema for a database table with specified vector size."""
    from lancedb.pydantic import LanceModel


    class Schema(LanceModel):
        idx: int
        im_file: str
        count: int
        sim_im_files: List[str]
    return Schema
