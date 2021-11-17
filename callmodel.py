import models
import torch
model = models.getModel(1)
img = torch.randn(10, 3, 32, 32)
out = models.predict(model,img)
print(out.shape)