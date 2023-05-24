from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")  # 创建示例，内容保存到logs文件夹

writer.add_image()
for i in range(1, 100):
    writer.add_scalar("y = x", i, i)

writer.close()
