from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model ---------------- #
        self.num_classes = 3
        self.class_names = ("ball", "human", "rim")
        self.depth = 0.33
        self.width = 0.25            # YOLOX-Nano
        self.act = "silu"

        # ---------------- dataset ---------------- #
        self.data_dir = "datasets/basketball"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.input_size = (736, 736)
        self.test_size  = (736, 736)

        # ---------------- training ---------------- #
        self.max_epoch = 300          # Nano often benefits from a bit more
        self.data_num_workers = 4

        # Nano is lighter: you can usually raise batch
        self.batch_size = 32          # try 16; if fine try 20/24; if OOM drop to 12
        self.eval_interval = 5
        self.no_aug_epochs = 15

        # Keep LR stable vs batch changes
        self.basic_lr_per_img = 0.01 / 64.0

        # ---------------- augmentation ---------------- #
        self.mosaic_prob = 0.8
        self.mixup_prob  = 0.2
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # ---------------- misc ---------------- #
        self.exp_name = "yolox_nano_basketball"
