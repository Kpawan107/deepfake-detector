from insightface.model_zoo import model_zoo

class FaceAlignModelLoader:
    def __init__(self, model_dir, category, model_name):
        self.model_dir = model_dir
        self.category = category
        self.model_name = model_name

    def load_model(self):
        model = model_zoo.get_model(self.model_name, download=True)
        return model, {"model_name": self.model_name}