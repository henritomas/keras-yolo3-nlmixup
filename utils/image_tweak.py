import cv2
import numpy as np
import copy

def random_paste(img1, img2, resize=0.35):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
  
    img2 = cv2.resize(img2, (int(w1*resize), int(h1*resize))) #w,h pag resize
    h2, w2 = img2.shape[:2]

    if h2>h1 or w2>w1:
        print("Error: resized source image still larger then destination")
        return None

    paste_range_h = h1-h2
    paste_range_w = w1-w2
    paste_h = np.random.randint(0, paste_range_h+1)
    paste_w = np.random.randint(0, paste_range_w+1)

    img1[paste_h:paste_h+h2,paste_w:paste_w+w2] = img2

    xmin = paste_w
    ymin = paste_h
    xmax = paste_w+w2
    ymax = paste_h+h2

    bbox = (xmin, ymin, xmax, ymax)

    return img1, bbox

def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1./scale;

def _constrain(min_v, max_v, value):
    if value < min_v: return min_v
    if value > max_v: return max_v
    return value 

def random_flip(image, flip):
    if flip == 1: return cv2.flip(image, 1)
    return image

def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1:
            swap = boxes[i]['xmin'];
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation);
    dexp = _rand_scale(exposure);     

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    
    # change satuation and exposure
    image[:,:,1] *= dsat
    image[:,:,2] *= dexp
    
    # change hue
    image[:,:,0] += dhue
    image[:,:,0] -= (image[:,:,0] > 180)*180
    image[:,:,0] += (image[:,:,0] < 0)  *180
    
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))
    
    if dx > 0: 
        im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:,-dx:,:]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)
               
    if dy > 0: 
        im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:,:,:]
        
    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)
        
    return im_sized[:net_h, :net_w,:]     