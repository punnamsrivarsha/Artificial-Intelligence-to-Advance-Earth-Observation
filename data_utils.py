import torch.nn.functional as F

def custom_collate_fn(batch):
    """
    Handles variable-sized patches by padding them to the size 
    of the largest patch in the current batch.
    """
    images, masks = zip(*batch)
    
    # Determine the max height and width in this batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    padded_images = []
    padded_masks = []
    
    for img, mask in zip(images, masks):
        # Calculate padding: (left, right, top, bottom)
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        # Pad image (C, H, W) and mask (H, W)
        padded_images.append(F.pad(img, (0, pad_w, 0, pad_h), value=0))
        padded_masks.append(F.pad(mask, (0, pad_w, 0, pad_h), value=0))
        
    return torch.stack(padded_images), torch.stack(padded_masks)

class AugmentationPipeline:
    """
    Encapsulates random transformations as a callable class.
    """
    def __call__(self, image, mask):
        # Example: Random Horizontal Flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        return image, mask