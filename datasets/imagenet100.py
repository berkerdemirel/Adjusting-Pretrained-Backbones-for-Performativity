from torchvision import datasets

class ImageNet100(datasets.ImageFolder):
    def __init__(
        self, 
        root: str,
        transform=None,
        target_transform=None,
    ):
        
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.num_classes = 100


    @staticmethod
    def domains():
        return [
            "none"
        ]
        