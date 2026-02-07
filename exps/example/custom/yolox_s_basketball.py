from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model ---------------- #
        self.num_classes = 3
        self.class_names = ("ball", "human", "rim")
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        # ---------------- dataset ---------------- #
        self.data_dir = "datasets/basketball"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.input_size = (736, 736)
        self.test_size  = (736, 736)

        # ---------------- training ---------------- #
        self.max_epoch = 150
        self.data_num_workers = 4

        # GTX 1070 safe values
        self.batch_size = 8
        self.eval_interval = 5

        # Refinement phase at end
        self.no_aug_epochs = 15

        # ---------------- augmentation ---------------- #
        self.mosaic_prob = 0.8
        self.mixup_prob  = 0.2
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # ---------------- misc ---------------- #
        self.exp_name = "yolox_s_basketball"
