from PIL                            import Image
from tensorflow.keras.preprocessing import image

import numpy             as np
import pandas            as pd
import os 
import random


def convert_image(filepath, extra_top_bottom, target_size, data_augmentation = False):
    img = Image.open(filepath).convert('RGBA')
    original_width, original_height = img.size

    max_dim     = max(original_width, original_height)
    square_img  = Image.new('RGBA', (max_dim, max_dim), (255, 255, 255, 255))
    offset_x    = (max_dim - original_width) // 2
    offset_y    = (max_dim - original_height) // 2
    square_img.paste(img, (offset_x, offset_y), img)

    # Add white stripe at top and bottom
    padded_height = max_dim + 2 * extra_top_bottom
    padded_img    = Image.new('RGBA', (max_dim, padded_height), (255, 255, 255, 255))
    padded_img.paste(square_img, (0, extra_top_bottom), square_img)

    # Resize to final size
    resized_img = padded_img.resize(target_size, Image.Resampling.LANCZOS)
    if data_augmentation:
        return resized_img
    else:
        img       = resized_img.convert('RGB')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        return img_array
    

def dataseparation(
    input_dir            = 'data',
    target_size          = (256, 256),
    k                    = 4,
    REPRODUCIBILITY_SEED = 60,
    val_split            = 0.15,
    angles               = [12 * j for j in range(30)]
    ):
    
    extra_top_bottom     = 10
    
    REPRODUCIBILITY_SEED = 60
    random.seed(REPRODUCIBILITY_SEED)
    
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif'))]
    
    random.shuffle(all_files)

    num_total = len(all_files)
    num_test_val = int(val_split * num_total)
    
    i = k-1
        
    start_idx_val   = i * num_test_val
    end_idx_val     = (i + 1) * num_test_val
    start_idx_test  = (i + 1) * num_test_val
    end_idx_test    = (i + 2) * num_test_val

    if end_idx_test < len(all_files):
        val     = all_files[start_idx_val:end_idx_val]
        test    = all_files[start_idx_test:end_idx_test]
        train   = all_files[:start_idx_val] + all_files[end_idx_test:]

        # 1. Create a list of dictionaries to organize the data row-by-row
        csv_data = []

        for i in range(len(all_files)):
            if all_files[i] in train:
                csv_data.append({f'Train_{k}': all_files[i], f'Val_{k}': "-", f'Test_{k}': "-"})
            elif all_files[i] in val:
                csv_data.append({f'Train_{k}': "-", f'Val_{k}': all_files[i], f'Test_{k}': "-"})
            elif all_files[i] in test:
                csv_data.append({f'Train_{k}': "-", f'Val_{k}': "-", f'Test_{k}': all_files[i]})

        # 2. Convert the list into a structured Pandas DataFrame
        df = pd.DataFrame(csv_data)

        # 3. Save to a CSV file
        csv_filename = f"./metrics/data_split_fold_{k}.csv"
        df.to_csv(csv_filename, index=False)

        x_val_data    = []
        x_test_data   = []
        x_train_data  = []

        y_val_data    = []
        y_test_data   = []
        y_train_data  = []
        
        for filename in val:
            filepath    = os.path.join(input_dir, filename)
            img_array   = convert_image(filepath, extra_top_bottom, target_size)
            x_val_data.append(img_array)
            if 'explosive' in filename:
                y_val_data.append(1)
            else:
                y_val_data.append(0)
            
        for filename in test:
            filepath  = os.path.join(input_dir, filename)
            img_array = convert_image(filepath, extra_top_bottom, target_size)
            x_test_data.append(img_array)
            if 'explosive' in filename:
                y_test_data.append(1)
            else:
                y_test_data.append(0)
        
        for filename in train:
            filepath = os.path.join(input_dir, filename)
            for i, angle in enumerate(angles, start=1):
                resized_img = convert_image(filepath, extra_top_bottom, target_size, data_augmentation=True)
                
                canvas      = Image.new('RGBA', target_size, (255, 255, 255, 255))
                canvas.paste(resized_img, (0, 0), resized_img)

                rotated     = canvas.rotate(angle, expand=False)
                
                white_bg    = Image.new('RGBA', rotated.size, (255, 255, 255, 255))
                final_rgba  = Image.alpha_composite(white_bg, rotated)

                img         = final_rgba.convert('RGB')
                img_array   = image.img_to_array(img)
                img_array   = img_array / 255.0

                x_train_data.append(img_array)
                if 'explosive' in filename:
                    y_train_data.append(1)
                else:
                    y_train_data.append(0)
        
        x_val_data      = np.array(x_val_data)
        y_val_data      = np.array(y_val_data)

        x_test_data     = np.array(x_test_data)
        y_test_data     = np.array(y_test_data)

        x_train_data    = np.array(x_train_data)
        y_train_data    = np.array(y_train_data)
        
        return x_val_data, x_test_data, x_train_data, y_val_data, y_test_data, y_train_data
    else:
        print(f"The number of folds exceeds the {val_split*100} percentile for the test and validation group.")
        
#dataseparation(target_size=(128,128),k=6)