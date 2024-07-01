import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ------------------------- Visualize ------------------------- # 

def visualize_layout(
    layout: list,
    width: float = 1.0,
    height: float = 1.0,
    save_dir: str = 'visual/',
):
    '''
    Function:
        visualize layout
    '''
    
    # set image
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, width * 1.1)
    ax.set_ylim(0, height * 1.1)
    
    # set boundary
    rect = patches.Rectangle((0, 0), width, height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # draw layout
    for obj in layout:
        
        name = obj['object']
        bbox = obj['bbox']
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none'
        )
        
        # add to image
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], name, color='r', fontsize=8, verticalalignment='top')
    
    plt.axis('off')
    
    # invert y axis
    plt.gca().invert_yaxis()
    
    # save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cnt = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
    plt.savefig(f"{save_dir}{cnt}.png")
    
    plt.show()


def visualize_seg(syn_data: dict, save_dir: str = 'visual/'):
    '''
    Function:
        visualize segmentation
    '''
    
    image = cv2.imread(syn_data['img_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i in syn_data['layout']:
        contour = np.array(i['segmentation'][0]).reshape(-1, 2).astype(np.int32)
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
    
    # plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    # save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cnt = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
    
    plt.savefig(f"{save_dir}{cnt}.png", bbox_inches='tight', pad_inches=0)
    