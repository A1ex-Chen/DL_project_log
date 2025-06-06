from .Singleturn import Direct3D_Inferencer, Direct_Inferencer, PPL_Inferencer, Trace_Inferencer
from .Multiturn import Multi_Turn_PPL_Inferencer, Multi_Direct_Inferencer

inferencer_dict = {
    'Direct': Direct_Inferencer,
    'Direct3D': Direct3D_Inferencer,  
    'PPL': PPL_Inferencer,
    'Multi_PPL': Multi_Turn_PPL_Inferencer,
    'Multi_Direct': Multi_Direct_Inferencer,
    'Trace': Trace_Inferencer,
}
