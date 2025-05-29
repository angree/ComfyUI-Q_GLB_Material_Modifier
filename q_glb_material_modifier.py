import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch
from io import BytesIO
import base64

class QManualGLBMaterialModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "multiline": False}),
                "texture_image": ("IMAGE", {"forceInput": True}),
                "output_suffix": ("STRING", {"default": "_modified", "multiline": False}),
                "metallic_factor": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smoothness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_color_r": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_color_g": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_color_b": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emissive_brightness_threshold": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01}),
                "emissive_percentage": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.15, "step": 0.01}),
                "absolute_min_brightness": ("INT", {"default": 128, "min": 50, "max": 200, "step": 1}),
                "emissive_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "emissive_mode": (["No Emissive", "With Emissive", "Both Versions"], {"default": "With Emissive"}),
                "render": ("BOOLEAN", {"default": True}),
                "normal_mode_1": (["No Normal Map", "Use Input Texture", "AutoGen Subtle", "AutoGen Normal", "AutoGen Enhanced", "AutoGen Dramatic", "AutoGen Smooth"], {"default": "No Normal Map"}),
                "normal_mode_2": (["Disable", "No Normal Map", "Use Input Texture", "AutoGen Subtle", "AutoGen Normal", "AutoGen Enhanced", "AutoGen Dramatic", "AutoGen Smooth"], {"default": "Disable"}),
                "normal_algorithm": (["Sobel Filter", "Simple Gradient"], {"default": "Sobel Filter"}),
                "normal_scale": (["1x (Original)", "2x (Half Size)", "4x (Quarter Size)"], {"default": "1x (Original)"}),
                "normal_compression": ("BOOLEAN", {"default": True}),
                "normal_noise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
            "optional": {
                "normal_texture": ("IMAGE", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("modified_glb_path", "emissive_mask_preview", "normal_map_preview")
    FUNCTION = "modify_material_smart"
    CATEGORY = "3d/q_material"

    def create_emissive_texture(self, original_image, emissive_mask, emissive_strength):
        """Creates a proper colored emissive texture that preserves original colors in emissive areas"""
        # Convert mask to 3-channel if needed
        if len(emissive_mask.shape) == 2:
            emissive_mask_3d = np.stack([emissive_mask, emissive_mask, emissive_mask], axis=-1)
        else:
            emissive_mask_3d = emissive_mask
            
        # Normalize mask to [0,1]
        mask_normalized = emissive_mask_3d.astype(np.float32) / 255.0
        
        # Create colored emissive texture that preserves original colors in bright areas
        emissive_texture = original_image * mask_normalized * emissive_strength
        
        # Clip to valid range
        emissive_texture = np.clip(emissive_texture, 0, 1)
        
        return emissive_texture

    def analyze_texture_brightness(self, image_tensor, brightness_threshold, percentage, absolute_min_threshold=128):
        """Analyzes texture and creates smart emissive mask"""
        # Check if it's a tensor or already a numpy array
        if hasattr(image_tensor, 'cpu'):
            image_np_raw = image_tensor.cpu().numpy()
        else:
            image_np_raw = image_tensor
            
        # Remove batch dimension if it exists
        if len(image_np_raw.shape) == 4:
            image_np_raw = image_np_raw[0]
            
        # Convert from float [0,1] to uint8 [0,255]
        image_np = (image_np_raw * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(image_np)
        
        # Convert to grayscale for brightness analysis
        gray = pil_image.convert('L')
        gray_np = np.array(gray)
        
        # Calculate brightness statistics
        mean_brightness = np.mean(gray_np) / 255.0
        
        # Find percentile for brightest pixels
        bright_threshold_value = np.percentile(gray_np, (1.0 - percentage) * 100)
        
        # Use higher value from two thresholds
        final_threshold = max(bright_threshold_value, absolute_min_threshold)
        
        # Check if brightest pixels are significantly brighter than average
        brightness_contrast = (final_threshold / 255.0) - mean_brightness
        
        # Create emissive mask only if contrast is sufficient
        if brightness_contrast >= brightness_threshold:
            # Create mask for brightest pixels
            emissive_mask = (gray_np >= final_threshold).astype(np.uint8) * 255
            
            # Optional: slight blur of mask for smoother transition
            emissive_mask = cv2.GaussianBlur(emissive_mask, (3, 3), 0)
            
            # Normalize mask
            if np.max(emissive_mask) > 0:
                emissive_mask = (emissive_mask / np.max(emissive_mask) * 255).astype(np.uint8)
            else:
                emissive_mask = np.zeros_like(gray_np, dtype=np.uint8)
                return emissive_mask, False
                
            return emissive_mask, True
        else:
            return np.zeros_like(gray_np, dtype=np.uint8), False

    def image_to_data_uri(self, image_array, format='png', texture_type='color'):
        """Convert image array to data URI with optimized compression"""
        if len(image_array.shape) == 3:
            # Convert to PIL Image
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                # Convert float [0,1] to uint8 [0,255]
                image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            
            # Save to BytesIO with optimized settings
            buffer = BytesIO()
            
            if format.lower() == 'png':
                # Optimize PNG compression based on texture type
                if texture_type == 'normal':
                    pil_image.save(buffer, format='PNG', optimize=True, compress_level=9)
                elif texture_type == 'emissive':
                    pil_image.save(buffer, format='PNG', optimize=True, compress_level=6)
                else:
                    pil_image.save(buffer, format='PNG', optimize=True, compress_level=6)
            elif format.lower() == 'jpeg':
                pil_image.save(buffer, format='JPEG', optimize=True, quality=85)
            else:
                pil_image.save(buffer, format=format)
            
            buffer.seek(0)
            
            # Convert to base64
            mime_type = f"image/{format.lower()}"
            encoded = base64.b64encode(buffer.read()).decode('ascii')
            return f"data:{mime_type};base64,{encoded}"
        return None

    def get_normal_mode_suffix(self, normal_mode):
        """Get filename suffix for normal mode"""
        suffix_map = {
            "No Normal Map": "_nonormal",
            "Use Input Texture": "_inputnormal", 
            "AutoGen Subtle": "_normalautogen_sub",
            "AutoGen Normal": "_normalautogen_nor",
            "AutoGen Enhanced": "_normalautogen_enc",
            "AutoGen Dramatic": "_normalautogen_dra",
            "AutoGen Smooth": "_normalautogen_smo",
            "Disable": ""  # No suffix for disabled
        }
        return suffix_map.get(normal_mode, "_nonormal")

    def get_normal_preset_params(self, preset_name):
        """Get parameters for normal map generation presets"""
        presets = {
            "AutoGen Subtle": {
                "strength": 0.7,
                "blur_radius": 1.0,
                "contrast": 1.0,
                "detail_boost": 1.0
            },
            "AutoGen Normal": {
                "strength": 1.0,
                "blur_radius": 0.5,
                "contrast": 1.0,
                "detail_boost": 1.0
            },
            "AutoGen Enhanced": {
                "strength": 1.3,
                "blur_radius": 0.3,
                "contrast": 1.2,
                "detail_boost": 1.2
            },
            "AutoGen Dramatic": {
                "strength": 1.8,
                "blur_radius": 0.1,
                "contrast": 1.5,
                "detail_boost": 1.5
            },
            "AutoGen Smooth": {
                "strength": 1.0,
                "blur_radius": 2.0,
                "contrast": 0.8,
                "detail_boost": 0.8
            }
        }
        return presets.get(preset_name, presets["AutoGen Normal"])

    def resize_texture(self, texture_array, scale_option):
        """Resize texture based on scale option"""
        scale_factor = 1
        if scale_option == "2x (Half Size)":
            scale_factor = 2
        elif scale_option == "4x (Quarter Size)":
            scale_factor = 4
        
        if scale_factor == 1:
            return texture_array
        
        # Calculate new dimensions
        if len(texture_array.shape) == 3:
            height, width, channels = texture_array.shape
            new_height = height // scale_factor
            new_width = width // scale_factor
            
            # Use PIL for high-quality resizing
            from PIL import Image
            pil_image = Image.fromarray(texture_array)
            resized_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            return np.array(resized_pil)
        else:
            return texture_array

    def generate_normal_from_diffuse(self, diffuse_texture, method="Sobel Filter", strength=1.0, 
                                blur_radius=0.5, contrast=1.0, detail_boost=1.0, scale_option="1x (Original)"):
        # Konwersja tensora na numpy, jeśli potrzebne
        if hasattr(diffuse_texture, 'cpu'):
            img_np = diffuse_texture.cpu().numpy()
        else:
            img_np = diffuse_texture
            
        # Usunięcie wymiaru batch, jeśli istnieje
        if len(img_np.shape) == 4:
            img_np = img_np[0]
        
        # Upewnienie się, że dane są w zakresie [0,1] typu float
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float64) / 255.0
        
        # Konwersja na mapę wysokości (grayscale) przy użyciu formuły luminancji
        if len(img_np.shape) == 3:
            heightmap = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
        else:
            heightmap = img_np
        
        # Normalizacja mapy wysokości do [0,1]
        height_min = np.min(heightmap)
        height_max = np.max(heightmap)
        if height_max > height_min:
            heightmap = (heightmap - height_min) / (height_max - height_min)
        else:
            heightmap = np.zeros_like(heightmap)
        
        # Opcjonalne przetwarzanie wstępne
        if blur_radius > 0:
            sigma = blur_radius
            heightmap = cv2.GaussianBlur(heightmap, (0, 0), sigma)
        
        # Wzmacnianie kontrastu, jeśli podano
        if contrast != 1.0:
            heightmap = np.clip((heightmap - 0.5) * contrast + 0.5, 0, 1)
        
        height, width = heightmap.shape
        
        print(f"Generowanie mapy normalnych metodą {method}")
        
        if method == "Sobel Filter":
            # Obliczanie gradientów na mapie wysokości typu float
            grad_x = cv2.Sobel(heightmap, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(heightmap, cv2.CV_64F, 0, 1, ksize=3)
            grad_x *= strength
            grad_y *= strength
            normal_x = -grad_x
            normal_y = -grad_y
            normal_z = np.ones_like(normal_x)
        
        elif method == "Simple Gradient":
            normal_x = np.zeros_like(heightmap)
            normal_y = np.zeros_like(heightmap)
            normal_z = np.ones_like(heightmap)
            for y in range(1, height-1):
                for x in range(1, width-1):
                    h_left = heightmap[y, x-1]
                    h_right = heightmap[y, x+1]
                    h_up = heightmap[y-1, x]
                    h_down = heightmap[y+1, x]
                    dx = (h_right - h_left) / 2.0 * strength
                    dy = (h_down - h_up) / 2.0 * strength
                    normal_x[y, x] = -dx
                    normal_y[y, x] = -dy
        
        # Normalizacja wektorów normalnych
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        length = np.maximum(length, 1e-8)  # Unikanie dzielenia przez zero
        normal_x /= length
        normal_y /= length
        normal_z /= length
        
        # Konwersja z [-1,1] na [0,1] dla zapisu
        normal_x = (normal_x + 1.0) * 0.5
        normal_y = (normal_y + 1.0) * 0.5
        normal_z = (normal_z + 1.0) * 0.5
        
        # Upewnienie się, że kanał Z jest odpowiedni (> 0.5 dla normalnych skierowanych na zewnątrz)
        normal_z = np.maximum(normal_z, 0.5)
        
        # Połączenie w obraz RGB
        normal_map = np.stack([normal_x, normal_y, normal_z], axis=2)
        
        # Konwersja na 8-bitowy format
        normal_map_8bit = (normal_map * 255).astype(np.uint8)
        
        print(f"Wygenerowano mapę normalnych {width}x{height} metodą {method}")
        
        return normal_map_8bit

    def apply_normal_noise(self, normal_texture, noise_strength):
        """Apply procedural noise to normal map for surface roughness/detail"""
        if noise_strength <= 0:
            return normal_texture
            
        # Ensure we're working with numpy array
        if hasattr(normal_texture, 'cpu'):
            normal_np = normal_texture.cpu().numpy()
        else:
            normal_np = normal_texture.copy()
        
        # Ensure float32 for processing
        if normal_np.dtype == np.uint8:
            normal_np = normal_np.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            was_uint8 = False
            
        height, width = normal_np.shape[:2]
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate multi-octave Perlin-like noise using OpenCV
        # We'll combine multiple scales for more natural detail
        noise_map = np.zeros((height, width), dtype=np.float32)
        
        # Octave 1: Large scale features
        scale1 = 50.0
        noise1 = np.random.rand(int(height/scale1) + 2, int(width/scale1) + 2)
        noise1 = cv2.resize(noise1, (width, height), interpolation=cv2.INTER_CUBIC)
        noise1 = cv2.GaussianBlur(noise1, (0, 0), scale1/4)
        noise_map += noise1 * 0.5
        
        # Octave 2: Medium details
        scale2 = 20.0
        noise2 = np.random.rand(int(height/scale2) + 2, int(width/scale2) + 2)
        noise2 = cv2.resize(noise2, (width, height), interpolation=cv2.INTER_CUBIC)
        noise2 = cv2.GaussianBlur(noise2, (0, 0), scale2/4)
        noise_map += noise2 * 0.3
        
        # Octave 3: Fine details
        scale3 = 10.0
        noise3 = np.random.rand(int(height/scale3) + 2, int(width/scale3) + 2)
        noise3 = cv2.resize(noise3, (width, height), interpolation=cv2.INTER_CUBIC)
        noise3 = cv2.GaussianBlur(noise3, (0, 0), scale3/4)
        noise_map += noise3 * 0.2
        
        # Normalize noise to [-1, 1]
        noise_map = (noise_map - 0.5) * 2.0
        
        # Apply noise to normal map
        # Convert normal map from [0,1] to [-1,1]
        normal_vectors = normal_np * 2.0 - 1.0
        
        # Apply noise as perturbation to X and Y channels
        # Noise strength controls how much the normals are perturbed
        normal_vectors[:, :, 0] += noise_map * noise_strength * 0.5  # X (red)
        normal_vectors[:, :, 1] += noise_map * noise_strength * 0.5  # Y (green)
        
        # Apply slight noise to Z channel too, but much less
        z_noise = cv2.GaussianBlur(np.random.rand(height, width), (0, 0), 5) - 0.5
        normal_vectors[:, :, 2] += z_noise * noise_strength * 0.1  # Z (blue)
        
        # Renormalize the vectors
        length = np.sqrt(np.sum(normal_vectors**2, axis=2, keepdims=True))
        length = np.maximum(length, 1e-8)
        normal_vectors /= length
        
        # Convert back to [0,1] range
        normal_np = (normal_vectors + 1.0) * 0.5
        
        # Ensure Z is still pointing outward (> 0.5)
        normal_np[:, :, 2] = np.maximum(normal_np[:, :, 2], 0.5)
        
        # Convert back to original dtype if needed
        if was_uint8:
            normal_np = (normal_np * 255).astype(np.uint8)
            
        return normal_np

    def process_normal_texture(self, normal_texture, enable_compression=True):
        """Process normal texture to ensure it's in the correct format for glTF"""
        # Check if it's a tensor or already a numpy array
        if hasattr(normal_texture, 'cpu'):
            normal_np = normal_texture.cpu().numpy()
        else:
            normal_np = normal_texture
            
        # Remove batch dimension if it exists
        if len(normal_np.shape) == 4:
            normal_np = normal_np[0]
        
        # Ensure it's in [0,1] range for processing
        if normal_np.dtype == np.uint8:
            normal_np = normal_np.astype(np.float32) / 255.0
        
        # If the image is grayscale, convert to a flat normal map
        if len(normal_np.shape) == 2 or (len(normal_np.shape) == 3 and normal_np.shape[2] == 1):
            height, width = normal_np.shape[:2]
            flat_normal = np.zeros((height, width, 3), dtype=np.float32)
            flat_normal[:, :, 0] = 0.5  # X = 0 (neutral)
            flat_normal[:, :, 1] = 0.5  # Y = 0 (neutral)  
            flat_normal[:, :, 2] = 1.0  # Z = 1 (pointing up)
            normal_np = flat_normal
        
        # Ensure we have exactly 3 channels
        if normal_np.shape[2] > 3:
            normal_np = normal_np[:, :, :3]
        
        # SIMPLIFIED RG compression (only if enabled)
        if enable_compression:
            print("Applying light RG optimization to normal map")
            
            # Ensure blue channel (Z) is reasonable for normal maps
            blue_channel = normal_np[:, :, 2]
            
            # If blue channel is too low, boost it slightly
            min_blue = np.min(blue_channel)
            if min_blue < 0.3:
                normal_np[:, :, 2] = np.maximum(blue_channel, 0.5)
                print(f"Boosted blue channel from min {min_blue:.3f} to ensure proper normal mapping")
        else:
            print("RG compression disabled - using original normal map data")
        
        # Convert to 8-bit for saving
        normal_8bit = (normal_np * 255).astype(np.uint8)
        
        return normal_8bit

    def process_single_normal_mode(self, normal_mode, texture_image, normal_texture, 
                                 normal_algorithm, normal_scale, normal_compression, normal_noise):
        """Process a single normal mode and return the data URI and suffix"""
        if normal_mode == "Disable" or normal_mode == "No Normal Map":
            return None, False, None
            
        normal_data_uri = None
        has_normal = False
        normal_preview = None
        
        print(f"Processing normal mode: {normal_mode}")
        
        if normal_mode == "Use Input Texture" and normal_texture is not None:
            print("Processing provided normal texture")
            try:
                processed_normal = self.process_normal_texture(normal_texture, normal_compression)
                
                # Apply scaling if requested
                if normal_scale != "1x (Original)":
                    print(f"Scaling normal texture: {normal_scale}")
                    processed_normal = self.resize_texture(processed_normal, normal_scale)
                
                # Apply noise if requested
                if normal_noise > 0:
                    print(f"Applying normal noise with strength: {normal_noise}")
                    processed_normal = self.apply_normal_noise(processed_normal, normal_noise)
                
                print(f"Processed normal shape: {processed_normal.shape}")
                normal_data_uri = self.image_to_data_uri(processed_normal, texture_type='normal')
                has_normal = True
                
                # Create preview
                if processed_normal.dtype == np.uint8:
                    normal_preview = processed_normal.astype(np.float32) / 255.0
                else:
                    normal_preview = processed_normal
                    
                print("Normal texture processed successfully")
            except Exception as e:
                print(f"Error processing normal texture: {e}")
                has_normal = False
                
        elif normal_mode.startswith("AutoGen"):
            print(f"Auto-generating normal map using preset: {normal_mode}")
            print(f"Algorithm: {normal_algorithm}")
            
            try:
                # Get preset parameters
                preset_params = self.get_normal_preset_params(normal_mode)
                print(f"Preset params: {preset_params}")
                
                # Generate normal map from diffuse texture
                generated_normal = self.generate_normal_from_diffuse(
                    texture_image, 
                    method=normal_algorithm,
                    strength=preset_params["strength"],
                    blur_radius=preset_params["blur_radius"],
                    contrast=preset_params["contrast"],
                    detail_boost=preset_params["detail_boost"],
                    scale_option=normal_scale
                )
                
                # Apply scaling if requested and not already done
                if normal_scale != "1x (Original)":
                    print(f"Scaling generated normal texture: {normal_scale}")
                    generated_normal = self.resize_texture(generated_normal, normal_scale)
                
                # Apply noise if requested
                if normal_noise > 0:
                    print(f"Applying normal noise with strength: {normal_noise}")
                    generated_normal = self.apply_normal_noise(generated_normal, normal_noise)
                
                print(f"Generated normal shape: {generated_normal.shape}")
                normal_data_uri = self.image_to_data_uri(generated_normal, texture_type='normal')
                has_normal = True
                
                # Create preview
                if generated_normal.dtype == np.uint8:
                    normal_preview = generated_normal.astype(np.float32) / 255.0
                else:
                    normal_preview = generated_normal
                    
                print("Normal map auto-generated successfully")
                
            except Exception as e:
                print(f"Error auto-generating normal map: {e}")
                has_normal = False
        
        return normal_data_uri, has_normal, normal_preview

    def modify_material_smart(self, glb_path, texture_image, output_suffix, metallic_factor, smoothness, 
                             base_color_r, base_color_g, base_color_b, 
                             emissive_brightness_threshold, emissive_percentage, absolute_min_brightness,
                             emissive_strength, emissive_mode, render, normal_mode_1, normal_mode_2, normal_algorithm, 
                             normal_scale, normal_compression, normal_noise, normal_texture=None):
        
        try:
            from pygltflib import GLTF2, TextureInfo, Image as GLTFImage, Texture, Sampler
        except ImportError:
            print("ERROR: pygltflib not installed. Install with: pip install pygltflib")
            black_mask = np.zeros((512, 512, 3), dtype=np.float32)
            flat_normal = np.zeros((512, 512, 3), dtype=np.float32)
            flat_normal[:, :, 0] = 0.5
            flat_normal[:, :, 1] = 0.5
            flat_normal[:, :, 2] = 1.0
            return ("PYGLTFLIB_NOT_INSTALLED", torch.from_numpy(black_mask[None, ...]), torch.from_numpy(flat_normal[None, ...]))
        
        # Check if rendering is disabled
        if not render:
            print("Rendering disabled, skipping GLB processing")
            if hasattr(texture_image, 'cpu'):
                shape = texture_image[0].cpu().numpy().shape
            else:
                shape = texture_image[0].shape if len(texture_image.shape) == 4 else texture_image.shape
            
            black_mask = np.zeros(shape, dtype=np.float32)
            flat_normal = np.zeros(shape, dtype=np.float32)
            flat_normal[:, :, 0] = 0.5
            flat_normal[:, :, 1] = 0.5
            flat_normal[:, :, 2] = 1.0
            return ("RENDER_DISABLED", torch.from_numpy(black_mask[None, ...]), torch.from_numpy(flat_normal[None, ...]))
            
        # Path validation and clean-up
        if not glb_path or glb_path.strip() == "":
            print("ERROR: No GLB file path provided")
            black_mask = np.zeros((512, 512, 3), dtype=np.float32)
            flat_normal = np.zeros((512, 512, 3), dtype=np.float32)
            flat_normal[:, :, 0] = 0.5
            flat_normal[:, :, 1] = 0.5
            flat_normal[:, :, 2] = 1.0
            return ("NO_PATH_PROVIDED", torch.from_numpy(black_mask[None, ...]), torch.from_numpy(flat_normal[None, ...]))
        
        # Clean up path - remove quotes and whitespace
        glb_path = glb_path.strip().strip('"\'')
        
        # Find the actual GLB file
        actual_glb_path = None
        if os.path.exists(glb_path) and os.path.isfile(glb_path):
            actual_glb_path = glb_path
        else:
            # Check various possible locations
            if os.path.dirname(glb_path) == "":
                possible_paths = [
                    os.path.join(os.getcwd(), glb_path),
                    os.path.join("output", glb_path),
                    os.path.join("ComfyUI", "output", glb_path),
                    os.path.join("..", "output", glb_path),
                ]
            else:
                normalized_path = os.path.normpath(glb_path)
                possible_paths = [normalized_path]
                if not normalized_path.lower().endswith('.glb'):
                    possible_paths.append(f"{normalized_path}.glb")
            
            # Check all possible locations
            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    actual_glb_path = path
                    break
            
            if not actual_glb_path:
                # Wait and try again
                import time
                time.sleep(2)
                
                for path in possible_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        actual_glb_path = path
                        break
                
                if not actual_glb_path:
                    print(f"File not found: {glb_path}")
                    black_preview = np.zeros((1, 512, 512, 3), dtype=np.float32)
                    flat_normal = np.zeros((1, 512, 512, 3), dtype=np.float32)
                    flat_normal[:, :, :, 0] = 0.5
                    flat_normal[:, :, :, 1] = 0.5
                    flat_normal[:, :, :, 2] = 1.0
                    return ("FILE_NOT_FOUND", torch.from_numpy(black_preview), torch.from_numpy(flat_normal))
        
        # Process emissive texture
        emissive_mask = None
        has_emissive = False
        emissive_data_uri = None
        emissive_preview = None
        
        print(f"Emissive mode: {emissive_mode}")
        
        if emissive_mode in ["With Emissive", "Both Versions"]:
            # Generate emissive mask
            emissive_mask, has_emissive = self.analyze_texture_brightness(
                texture_image, emissive_brightness_threshold, emissive_percentage, absolute_min_brightness
            )
            
            if has_emissive:
                print(f"Creating emissive texture")
                
                # Get original texture for color reference
                if hasattr(texture_image, 'cpu'):
                    original_rgb = texture_image[0].cpu().numpy()
                else:
                    original_rgb = texture_image[0] if len(texture_image.shape) == 4 else texture_image
                
                # Create colored emissive texture
                emissive_texture = self.create_emissive_texture(
                    original_rgb, emissive_mask, emissive_strength
                )
                
                # Convert to 8-bit for saving
                emissive_texture_8bit = (emissive_texture * 255).astype(np.uint8)
                
                # Convert emissive texture to data URI
                emissive_data_uri = self.image_to_data_uri(emissive_texture_8bit, texture_type='emissive')
                
                # Save for preview
                emissive_preview = emissive_texture[None, ...]  # Add batch dimension
            else:
                print(f"No emissive areas detected")
        
        # Process both normal modes
        normal_data_uri_1, has_normal_1, normal_preview_1 = self.process_single_normal_mode(
            normal_mode_1, texture_image, normal_texture, 
            normal_algorithm, normal_scale, normal_compression, normal_noise
        )
        
        normal_data_uri_2, has_normal_2, normal_preview_2 = self.process_single_normal_mode(
            normal_mode_2, texture_image, normal_texture, 
            normal_algorithm, normal_scale, normal_compression, normal_noise
        )
        
        # Generate files based on combinations
        output_files = []
        base_path, ext = os.path.splitext(actual_glb_path)
        
        if emissive_mode == "Both Versions":
            # Generate all combinations
            print("Generating both emissive and non-emissive versions")
            
            # With emissive
            if has_emissive:
                # Normal mode 1 + emissive
                normal_suffix_1 = self.get_normal_mode_suffix(normal_mode_1)
                em_suffix_1 = output_suffix + normal_suffix_1 + "_EM"
                em_file_1 = self.generate_single_glb(actual_glb_path, em_suffix_1, metallic_factor, smoothness,
                                                   base_color_r, base_color_g, base_color_b,
                                                   True, emissive_data_uri, has_normal_1, normal_data_uri_1)
                output_files.append(em_file_1)
                print(f"Generated emissive version with normal mode 1: {em_file_1}")
                
                # Normal mode 2 + emissive (if not disabled)
                if normal_mode_2 != "Disable":
                    normal_suffix_2 = self.get_normal_mode_suffix(normal_mode_2)
                    em_suffix_2 = output_suffix + normal_suffix_2 + "_EM"
                    em_file_2 = self.generate_single_glb(actual_glb_path, em_suffix_2, metallic_factor, smoothness,
                                                       base_color_r, base_color_g, base_color_b,
                                                       True, emissive_data_uri, has_normal_2, normal_data_uri_2)
                    output_files.append(em_file_2)
                    print(f"Generated emissive version with normal mode 2: {em_file_2}")
            
            # Without emissive
            # Normal mode 1 + no emissive
            normal_suffix_1 = self.get_normal_mode_suffix(normal_mode_1)
            noem_suffix_1 = output_suffix + normal_suffix_1 + "_noEM"
            noem_file_1 = self.generate_single_glb(actual_glb_path, noem_suffix_1, metallic_factor, smoothness,
                                                 base_color_r, base_color_g, base_color_b,
                                                 False, None, has_normal_1, normal_data_uri_1)
            output_files.append(noem_file_1)
            print(f"Generated non-emissive version with normal mode 1: {noem_file_1}")
            
            # Normal mode 2 + no emissive (if not disabled)
            if normal_mode_2 != "Disable":
                normal_suffix_2 = self.get_normal_mode_suffix(normal_mode_2)
                noem_suffix_2 = output_suffix + normal_suffix_2 + "_noEM"
                noem_file_2 = self.generate_single_glb(actual_glb_path, noem_suffix_2, metallic_factor, smoothness,
                                                     base_color_r, base_color_g, base_color_b,
                                                     False, None, has_normal_2, normal_data_uri_2)
                output_files.append(noem_file_2)
                print(f"Generated non-emissive version with normal mode 2: {noem_file_2}")
            
            main_output = output_files[0] if output_files else "ERROR"
            
        elif emissive_mode == "With Emissive":
            # Generate only emissive versions
            # Normal mode 1 + emissive
            normal_suffix_1 = self.get_normal_mode_suffix(normal_mode_1)
            em_suffix_1 = output_suffix + normal_suffix_1 + "_EM"
            main_output = self.generate_single_glb(actual_glb_path, em_suffix_1, metallic_factor, smoothness,
                                                 base_color_r, base_color_g, base_color_b,
                                                 has_emissive, emissive_data_uri, has_normal_1, normal_data_uri_1)
            print(f"Generated emissive version with normal mode 1: {main_output}")
            
            # Normal mode 2 + emissive (if not disabled)
            if normal_mode_2 != "Disable":
                normal_suffix_2 = self.get_normal_mode_suffix(normal_mode_2)
                em_suffix_2 = output_suffix + normal_suffix_2 + "_EM"
                em_file_2 = self.generate_single_glb(actual_glb_path, em_suffix_2, metallic_factor, smoothness,
                                                   base_color_r, base_color_g, base_color_b,
                                                   has_emissive, emissive_data_uri, has_normal_2, normal_data_uri_2)
                print(f"Generated emissive version with normal mode 2: {em_file_2}")
            
        else:  # "No Emissive"
            # Generate only non-emissive versions
            # Normal mode 1 + no emissive
            normal_suffix_1 = self.get_normal_mode_suffix(normal_mode_1)
            noem_suffix_1 = output_suffix + normal_suffix_1 + "_noEM"
            main_output = self.generate_single_glb(actual_glb_path, noem_suffix_1, metallic_factor, smoothness,
                                                 base_color_r, base_color_g, base_color_b,
                                                 False, None, has_normal_1, normal_data_uri_1)
            print(f"Generated non-emissive version with normal mode 1: {main_output}")
            
            # Normal mode 2 + no emissive (if not disabled)
            if normal_mode_2 != "Disable":
                normal_suffix_2 = self.get_normal_mode_suffix(normal_mode_2)
                noem_suffix_2 = output_suffix + normal_suffix_2 + "_noEM"
                noem_file_2 = self.generate_single_glb(actual_glb_path, noem_suffix_2, metallic_factor, smoothness,
                                                     base_color_r, base_color_g, base_color_b,
                                                     False, None, has_normal_2, normal_data_uri_2)
                print(f"Generated non-emissive version with normal mode 2: {noem_file_2}")
        
        # Ensure emissive_preview is always defined
        if emissive_preview is None:
            # Create a black preview image
            if hasattr(texture_image, 'cpu'):
                shape = texture_image[0].cpu().numpy().shape
            else:
                shape = texture_image[0].shape if len(texture_image.shape) == 4 else texture_image.shape
            emissive_preview = np.zeros(shape, dtype=np.float32)[None, ...]
        
        # Ensure normal_preview is always defined
        # Use normal_preview_1 as the main normal preview
        if normal_preview_1 is None:
            # Create a flat normal map preview
            if hasattr(texture_image, 'cpu'):
                shape = texture_image[0].cpu().numpy().shape
            else:
                shape = texture_image[0].shape if len(texture_image.shape) == 4 else texture_image.shape
            normal_preview = np.zeros(shape, dtype=np.float32)
            normal_preview[:, :, 0] = 0.5  # X
            normal_preview[:, :, 1] = 0.5  # Y
            normal_preview[:, :, 2] = 1.0  # Z
            normal_preview = normal_preview[None, ...]
        else:
            normal_preview = normal_preview_1[None, ...] if len(normal_preview_1.shape) == 3 else normal_preview_1
        
        # Return all three outputs
        return (main_output, torch.from_numpy(emissive_preview), torch.from_numpy(normal_preview))
    def generate_single_glb(self, source_path, suffix, metallic_factor, smoothness, 
                          base_color_r, base_color_g, base_color_b,
                          use_emissive, emissive_data_uri, use_normal, normal_data_uri):
        """Generate a single GLB file with specified parameters"""
        try:
            from pygltflib import GLTF2, TextureInfo, Image as GLTFImage, Texture, Sampler
        except ImportError:
            return "PYGLTFLIB_NOT_INSTALLED"
        
        # Create output path
        base_path, ext = os.path.splitext(source_path)
        output_path = f"{base_path}{suffix}{ext}"
        
        # Load the GLB file
        gltf = GLTF2().load(source_path)
        
        # Update materials in the GLTF
        roughness_factor = 1.0 - smoothness  # Convert Unity smoothness to glTF roughness
        
        # Add textures to glTF if needed
        emissive_texture_index = None
        normal_texture_index = None
        
        # Add emissive texture if requested
        if use_emissive and emissive_data_uri:
            # Create an image for the emissive texture
            emissive_image = GLTFImage()
            emissive_image.uri = emissive_data_uri
            emissive_image.mimeType = "image/png"
            emissive_image_index = len(gltf.images)
            gltf.images.append(emissive_image)
            
            # Make sure we have a sampler
            if not gltf.samplers:
                default_sampler = Sampler()
                default_sampler.magFilter = 9729  # LINEAR
                default_sampler.minFilter = 9987  # LINEAR_MIPMAP_LINEAR
                default_sampler.wrapS = 10497     # REPEAT
                default_sampler.wrapT = 10497     # REPEAT
                gltf.samplers.append(default_sampler)
            
            # Create a texture referencing the image
            emissive_texture = Texture()
            emissive_texture.source = emissive_image_index
            emissive_texture.sampler = 0  # Use first sampler
            emissive_texture_index = len(gltf.textures)
            gltf.textures.append(emissive_texture)
        
        # Add normal texture if available
        if use_normal and normal_data_uri:
            # Create an image for the normal texture
            normal_image = GLTFImage()
            normal_image.uri = normal_data_uri
            normal_image.mimeType = "image/png"
            normal_image_index = len(gltf.images)
            gltf.images.append(normal_image)
            
            # Make sure we have a sampler
            if not gltf.samplers:
                default_sampler = Sampler()
                default_sampler.magFilter = 9729  # LINEAR
                default_sampler.minFilter = 9987  # LINEAR_MIPMAP_LINEAR
                default_sampler.wrapS = 10497     # REPEAT
                default_sampler.wrapT = 10497     # REPEAT
                gltf.samplers.append(default_sampler)
            
            # Create a texture referencing the image
            normal_texture_obj = Texture()
            normal_texture_obj.source = normal_image_index
            normal_texture_obj.sampler = 0  # Use first sampler
            normal_texture_index = len(gltf.textures)
            gltf.textures.append(normal_texture_obj)
        
        # Update all materials
        for i, material in enumerate(gltf.materials):
            # Set base PBR properties
            if not hasattr(material, 'pbrMetallicRoughness'):
                from pygltflib import PbrMetallicRoughness
                material.pbrMetallicRoughness = PbrMetallicRoughness()
            
            material.pbrMetallicRoughness.baseColorFactor = [
                base_color_r, base_color_g, base_color_b, 1.0
            ]
            material.pbrMetallicRoughness.metallicFactor = metallic_factor
            material.pbrMetallicRoughness.roughnessFactor = roughness_factor
            
            # Set emissive properties
            if use_emissive and emissive_texture_index is not None:
                # Create a TextureInfo for the emissive texture
                emissive_texture_info = TextureInfo()
                emissive_texture_info.index = emissive_texture_index
                material.emissiveTexture = emissive_texture_info
                material.emissiveFactor = [1.0, 1.0, 1.0]
            else:
                # No emissive
                material.emissiveFactor = [0.0, 0.0, 0.0]
            
            # Set normal texture properties
            if use_normal and normal_texture_index is not None:
                # Create a TextureInfo for the normal texture
                normal_texture_info = TextureInfo()
                normal_texture_info.index = normal_texture_index
                normal_strength = 1.0
                
                if hasattr(normal_texture_info, 'scale'):
                    normal_texture_info.scale = normal_strength
                else:
                    normal_texture_info = {
                        "index": normal_texture_index,
                        "scale": normal_strength
                    }
                
                material.normalTexture = normal_texture_info
        
        # Save the modified GLB file
        gltf.save(output_path)
        
        # Return only the filename (not full path)
        return os.path.basename(output_path)


# Simplified presets for Unity-style with normal texture support
class QPresetGLBMaterialModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "multiline": False}),
                "texture_image": ("IMAGE", {"forceInput": True}),
                "preset": (["Spaceship Metal", "Brushed Steel", "Chrome Hull", "Chrome Hull Lite", "Titanium", "Combat Metal", "Alien Tech"], {"default": "Spaceship Metal"}),
                "output_suffix": ("STRING", {"default": "_modified", "multiline": False}),
                "emissive_mode": (["No Emissive", "With Emissive", "Both Versions"], {"default": "With Emissive"}),
                "absolute_min_brightness": ("INT", {"default": 128, "min": 50, "max": 200, "step": 1}),
                "render": ("BOOLEAN", {"default": True}),
                "normal_mode_1": (["No Normal Map", "Use Input Texture", "AutoGen Subtle", "AutoGen Normal", "AutoGen Enhanced", "AutoGen Dramatic", "AutoGen Smooth"], {"default": "No Normal Map"}),
                "normal_mode_2": (["Disable", "No Normal Map", "Use Input Texture", "AutoGen Subtle", "AutoGen Normal", "AutoGen Enhanced", "AutoGen Dramatic", "AutoGen Smooth"], {"default": "Disable"}),
                "normal_algorithm": (["Sobel Filter", "Simple Gradient"], {"default": "Sobel Filter"}),
                "normal_scale": (["1x (Original)", "2x (Half Size)", "4x (Quarter Size)"], {"default": "1x (Original)"}),
                "normal_compression": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "normal_texture": ("IMAGE", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("modified_glb_path", "emissive_mask_preview", "normal_map_preview")
    FUNCTION = "apply_material_preset"
    CATEGORY = "3d/q_material"

    def apply_material_preset(self, glb_path, texture_image, preset, output_suffix, emissive_mode, 
                            absolute_min_brightness, render, normal_mode_1, normal_mode_2, normal_algorithm, 
                            normal_scale, normal_compression, normal_texture=None):
        # Unity-style presets
        presets = {
            "Spaceship Metal": {
                "metallic": 0.85, "smoothness": 0.5, 
                "color": [0.7, 0.8, 0.9], "emissive_strength": 0.8,
                "brightness_threshold": 0.2, "percentage": 0.05,
                "normal_noise": 0.1
            },
            "Brushed Steel": {
                "metallic": 0.9, "smoothness": 0.3, 
                "color": [0.8, 0.8, 0.8], "emissive_strength": 0.6,
                "brightness_threshold": 0.25, "percentage": 0.03,
                "normal_noise": 0.25
            },
            "Chrome Hull": {
                "metallic": 1.0, "smoothness": 0.9, 
                "color": [0.95, 0.95, 0.95], "emissive_strength": 0.9,
                "brightness_threshold": 0.15, "percentage": 0.08,
                "normal_noise": 0.05
            },
            "Chrome Hull Lite": {
                "metallic": 1.0, "smoothness": 0.72, 
                "color": [0.95, 0.95, 0.95], "emissive_strength": 0.9,
                "brightness_threshold": 0.15, "percentage": 0.08,
                "normal_noise": 0.12
            },
            "Titanium": {
                "metallic": 0.8, "smoothness": 0.4, 
                "color": [0.6, 0.6, 0.7], "emissive_strength": 0.7,
                "brightness_threshold": 0.22, "percentage": 0.04,
                "normal_noise": 0.15
            },
            "Combat Metal": {
                "metallic": 0.7, "smoothness": 0.2, 
                "color": [0.5, 0.5, 0.6], "emissive_strength": 1.0,
                "brightness_threshold": 0.18, "percentage": 0.06,
                "normal_noise": 0.3
            },
            "Alien Tech": {
                "metallic": 0.6, "smoothness": 0.7, 
                "color": [0.4, 0.8, 0.6], "emissive_strength": 1.2,
                "brightness_threshold": 0.12, "percentage": 0.1,
                "normal_noise": 0.2
            }
        }
        
        settings = presets[preset]
        
        # Use the main node - pass all parameters including normal texture
        modifier = QManualGLBMaterialModifier()
        return modifier.modify_material_smart(
            glb_path=glb_path,
            texture_image=texture_image,
            output_suffix=output_suffix,
            metallic_factor=settings["metallic"],
            smoothness=settings["smoothness"],
            base_color_r=settings["color"][0],
            base_color_g=settings["color"][1],
            base_color_b=settings["color"][2],
            emissive_brightness_threshold=settings["brightness_threshold"],
            emissive_percentage=settings["percentage"],
            absolute_min_brightness=absolute_min_brightness,
            emissive_strength=settings["emissive_strength"],
            emissive_mode=emissive_mode,
            render=render,
            normal_mode_1=normal_mode_1,
            normal_mode_2=normal_mode_2,
            normal_algorithm=normal_algorithm,
            normal_scale=normal_scale,
            normal_compression=normal_compression,
            normal_noise=settings["normal_noise"],
            normal_texture=normal_texture
        )