import torchvision.transforms as T


def get_transforms(img_size, mode='train'):
    if mode == 'train':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

        ])
    elif mode == 'valid':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

        ])
    elif mode == 'test':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

        ])

def get_transforms_adv(img_size, mode='train'):
    if mode == 'train':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size))
        ])
    elif mode == 'valid':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size))

        ])
    elif mode == 'test':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size))
        ])
