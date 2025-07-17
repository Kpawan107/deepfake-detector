class FaceAlignModelHandler:
    def __init__(self, model, device, config):
        self.model = model
        self.model.prepare(ctx_id=0 if device.startswith("cuda") else -1)

    def inference_on_image(self, image, bbox):
        landmarks = self.model.get(image, bbox)
        return landmarks