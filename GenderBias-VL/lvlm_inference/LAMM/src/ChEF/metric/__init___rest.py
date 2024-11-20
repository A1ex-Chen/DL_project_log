from .vqa import VQA, MMBenchVQA, MMEVQA, LAMM_VQA

evaluation_protocol = {
    'basic':{
        'ScienceQA': VQA,
        'MMBench': MMBenchVQA,
        'MME': MMEVQA,
        'SEEDBench': VQA,
        'SEEDBench2': VQA,
        'OccBias': VQA,
    },
}
