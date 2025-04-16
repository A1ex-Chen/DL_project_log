def __init__(self, annotation_file=None, question_file=None):
    """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
    self.dataset = {}
    self.questions = {}
    self.qa = {}
    self.qqa = {}
    self.imgToQA = {}
    if not annotation_file == None and not question_file == None:
        print('loading VQA annotations and questions into memory...')
        time_t = datetime.datetime.utcnow()
        dataset = json.load(open(annotation_file, 'r'))
        questions = json.load(open(question_file, 'r'))
        self.dataset = dataset
        self.questions = questions
        self.createIndex()
