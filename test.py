from models.point_mae import MAEEncoder


model = MAEEncoder('/home/hty/PointContrast/outputs/tf/weights/weights.pth')
print(model.state_dict().keys())
