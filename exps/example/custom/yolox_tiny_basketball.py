from yolox.exp import Exp as MyExp

# & cmd /k '"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64'
# python .\scripts\train.py --exp exps/example/custom/yolox_tiny_basketball.py --base-name yolox_tiny_basketball --device gpu --batch 12 --ckpt pretrained/yolox_tiny.pth


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model ---------------- #
        self.num_classes = 3
        self.class_names = ("ball", "human", "rim")
        self.depth = 0.33
        self.width = 0.375          # YOLOX-Tiny
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

        # GTX 1070 safe values (you can probably increase batch with tiny)
        self.batch_size = 12        # try 12; if OOM drop to 8, if fine try 16
        self.eval_interval = 5
        self.no_aug_epochs = 15

        # IMPORTANT: scale LR with batch size (YOLOX does this via basic_lr_per_img)
        # Default is 0.01/64. Keep it explicit so changing batch doesn’t silently hurt.
        self.basic_lr_per_img = 0.01 / 64.0

        # ---------------- augmentation ---------------- #
        self.mosaic_prob = 0.8
        self.mixup_prob  = 0.2
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # ---------------- misc ---------------- #
        self.exp_name = "yolox_tiny_basketball"
