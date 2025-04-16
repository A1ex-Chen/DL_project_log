def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove=''
    ):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: boolean indicating whether TF2.0 and PyTorch weights matrices are transposed with regards to each
          other
    """
    tf_name = tf_name.replace(':0', '')
    tf_name = re.sub('/[^/]*___([^/]*)/', '/\\1/', tf_name)
    tf_name = tf_name.replace('_._', '/')
    tf_name = re.sub('//+', '/', tf_name)
    tf_name = tf_name.split('/')
    tf_name = tf_name[1:]
    transpose = bool(tf_name[-1] == 'kernel' or 'emb_projs' in tf_name or 
        'out_projs' in tf_name)
    if tf_name[-1] == 'kernel' or tf_name[-1] == 'embeddings' or tf_name[-1
        ] == 'gamma':
        tf_name[-1] = 'weight'
    if tf_name[-1] == 'beta':
        tf_name[-1] = 'bias'
    tf_name = '.'.join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, '', 1)
    return tf_name, transpose
