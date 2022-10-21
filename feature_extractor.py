class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    def get_features(self, agent):
        features = dict()

        features['my_pos'] = agent.pos

        return features
