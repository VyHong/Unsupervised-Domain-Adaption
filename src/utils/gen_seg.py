from PIL import Image

def convert_to_binary_map_pil(input_path, output_path, threshold_value=127):
    """
    Converts an image to a binary map using Pillow (PIL).

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the resulting binary map.
        threshold_value (int): Pixel value to separate black and white (0-255).
    """
    try:
        # 1. Open the image
        img = Image.open(input_path)

        # 2. Convert to Grayscale
        # The 'L' mode in Pillow represents a single-channel 8-bit image (grayscale).
        gray_img = img.convert('L')

        # 3. Apply Binary Thresholding
        # We define a function for the threshold mapping (0 or 255).
        def binarize(pixel):
            return 255 if pixel > threshold_value else 0

        # The point operation applies the binarize function to every pixel.
        binary_map = gray_img.point(binarize, mode='1')
        
        # Note: Using mode='1' (1-bit pixel, black and white) forces a true binary map, 
        # which is memory efficient. You can use 'L' if you prefer an 8-bit image 
        # that only contains values 0 and 255.

        # 4. Save the result
        binary_map.save(output_path)
        print(f"✅ Successfully converted '{input_path}' to binary map and saved at '{output_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Configuration ---
input_file = "data\drrs\drr01.png"    # <--- CHANGE THIS to your input file name
output_file = "data\segs\seg01.png" # <--- CHANGE THIS to your desired output file name
threshold = 120                   # <--- Adjust this value (0-255) to control the binarization

# Execute the function
convert_to_binary_map_pil(input_file, output_file, threshold)