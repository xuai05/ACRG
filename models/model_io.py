class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(self, state=None, hidden=None, detection_inputs=None, action_probs=None,state_name=None,scene_name=None,trainflag = True):
        self.state = state
        self.hidden = hidden
        self.detection_inputs = detection_inputs
        self.action_probs = action_probs
        # modify
        self.state_name = state_name
        self.scene_name = scene_name
        self.trainflag = True


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None):
        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
