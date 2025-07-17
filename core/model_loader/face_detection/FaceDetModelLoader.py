from insightface.model_zoo import model_zoo
import os

class FaceDetModelLoader:
    def __init__(self, model_dir, category, model_name):
        """
        Parameters:
        - model_dir (str): Directory to save/load models
        - category (str): Not used currently, but can be used for custom logic later
        - model_name (str): Must be a string like 'retinaface_mnet025_v2' or 'antelopev2'
        """
        self.model_dir = model_dir
        self.category = category
        self.model_name = model_name

    def load_model(self):
        if not isinstance(self.model_name, str):
            raise ValueError(f"Expected model_name to be a string, got: {type(self.model_name)}")

        try:
            model = model_zoo.get_model(self.model_name, root=os.path.abspath(self.model_dir), download=True)
            return model, {"model_name": self.model_name}
        except Exception as e:
            raise RuntimeError(f"Error loading model '{self.model_name}': {str(e)}")
