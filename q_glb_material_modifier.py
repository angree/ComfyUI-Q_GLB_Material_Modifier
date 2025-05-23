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
                "enable_smart_emissive": ("BOOLEAN", {"default": True}),
                "render": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("modified_glb_path", "emissive_mask_preview")
    FUNCTION = "modify_material_smart"
    CATEGORY = "3d/q_material"

    def create_emissive_texture(self, original_image, emissive_mask, emissive_strength):
        """
        Creates a proper colored emissive texture that preserves original colors in emissive areas
        """
        # Convert mask to 3-channel if needed
        if len(emissive_mask.shape) == 2:
            # Create a mask with 3 channels
            emissive_mask_3d = np.stack([emissive_mask, emissive_mask, emissive_mask], axis=-1)
        else:
            emissive_mask_3d = emissive_mask
            
        # Normalize mask to [0,1]
        mask_normalized = emissive_mask_3d.astype(np.float32) / 255.0
        
        # Create colored emissive texture that preserves original colors in bright areas
        # Use original colors where mask is bright, black where mask is dark
        emissive_texture = original_image * mask_normalized * emissive_strength
        
        # Clip to valid range
        emissive_texture = np.clip(emissive_texture, 0, 1)
        
        return emissive_texture

    def analyze_texture_brightness(self, image_tensor, brightness_threshold, percentage, absolute_min_threshold=128):
        """
        Analyzes texture and creates smart emissive mask
        """
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

    def image_to_data_uri(self, image_array, format='png'):
        """Convert image array to data URI"""
        if len(image_array.shape) == 3:
            # Convert to PIL Image
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                # Convert float [0,1] to uint8 [0,255]
                image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            
            # Save to BytesIO
            buffer = BytesIO()
            pil_image.save(buffer, format=format)
            buffer.seek(0)
            
            # Convert to base64
            mime_type = f"image/{format}"
            encoded = base64.b64encode(buffer.read()).decode('ascii')
            return f"data:{mime_type};base64,{encoded}"
        return None

    def modify_material_smart(self, glb_path, texture_image, output_suffix, metallic_factor, smoothness, 
                             base_color_r, base_color_g, base_color_b, 
                             emissive_brightness_threshold, emissive_percentage, absolute_min_brightness,
                             emissive_strength, enable_smart_emissive, render):
        
        try:
            from pygltflib import GLTF2, TextureInfo, Image as GLTFImage, Texture, Sampler
        except ImportError:
            print("ERROR: pygltflib not installed. Install with: pip install pygltflib")
            black_mask = np.zeros((512, 512, 3), dtype=np.float32)
            return ("PYGLTFLIB_NOT_INSTALLED", torch.from_numpy(black_mask[None, ...]))
        
        # Check if rendering is disabled
        if not render:
            print("Rendering disabled, skipping GLB processing")
            # Return only the filename and a black preview
            if hasattr(texture_image, 'cpu'):
                shape = texture_image[0].cpu().numpy().shape
            else:
                shape = texture_image[0].shape if len(texture_image.shape) == 4 else texture_image.shape
            
            black_mask = np.zeros(shape, dtype=np.float32)
            return ("RENDER_DISABLED", torch.from_numpy(black_mask[None, ...]))
            
        # Path validation and clean-up
        if not glb_path or glb_path.strip() == "":
            print("ERROR: No GLB file path provided")
            black_mask = np.zeros((512, 512, 3), dtype=np.float32)
            return ("NO_PATH_PROVIDED", torch.from_numpy(black_mask[None, ...]))
        
        # Clean up path - remove quotes and whitespace
        glb_path = glb_path.strip().strip('"\'')
        
        # First check if the path exists directly as provided
        if os.path.exists(glb_path) and os.path.isfile(glb_path):
            actual_glb_path = glb_path
        else:
            # Check if this is a relative path or just a filename
            if os.path.dirname(glb_path) == "":
                # This is just a filename, check in output directories
                possible_paths = [
                    os.path.join(os.getcwd(), glb_path),  # Current dir + filename
                    os.path.join("output", glb_path),  # output/ + filename  
                    os.path.join("ComfyUI", "output", glb_path),  # ComfyUI/output/ + filename
                    os.path.join("..", "output", glb_path),  # ../output/ + filename
                ]
            else:
                # This is a path with directories, just normalize it
                normalized_path = os.path.normpath(glb_path)
                possible_paths = [normalized_path]
                
                # Also try some variations with .glb extension if not already there
                if not normalized_path.lower().endswith('.glb'):
                    possible_paths.append(f"{normalized_path}.glb")
            
            # Check all possible locations
            found_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    found_path = path
                    break
            
            if found_path:
                actual_glb_path = found_path
            else:
                # Wait a bit and try again
                import time
                time.sleep(2)
                
                # Check again
                for path in possible_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        found_path = path
                        break
                
                if found_path:
                    actual_glb_path = found_path
                else:
                    print(f"File not found: {glb_path}")
                    black_preview = np.zeros((1, 512, 512, 3), dtype=np.float32)
                    return ("FILE_NOT_FOUND", torch.from_numpy(black_preview))
        
        # Create a new path for the modified file
        base_path, ext = os.path.splitext(actual_glb_path)
        modified_glb_path = f"{base_path}{output_suffix}{ext}"
        
        # Load the GLB file
        print(f"Loading GLB file: {actual_glb_path}")
        gltf = GLTF2().load(actual_glb_path)
        
        # Process emissive map
        emissive_mask = None
        has_emissive = False
        emissive_data_uri = None
        
        if enable_smart_emissive:
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
                emissive_data_uri = self.image_to_data_uri(emissive_texture_8bit)
                
                # Save for preview - use the colored emissive texture, not just the mask
                emissive_preview = emissive_texture[None, ...]  # Add batch dimension
            else:
                print(f"No emissive areas detected")
                emissive_preview = None
        else:
            print(f"Smart emissive disabled")
            has_emissive = False
            emissive_preview = None
        
        # Update materials in the GLTF
        roughness_factor = 1.0 - smoothness  # Convert Unity smoothness to glTF roughness
        
        # Add emissive texture to glTF if needed
        emissive_texture_index = None
        
        if has_emissive and emissive_data_uri:
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
            
            print(f"Added emissive texture at index {emissive_texture_index}")
        
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
            
            # Set emissive properties if we have them
            if has_emissive and emissive_texture_index is not None:
                # Create a TextureInfo for the emissive texture
                emissive_texture_info = TextureInfo()
                emissive_texture_info.index = emissive_texture_index
                material.emissiveTexture = emissive_texture_info
                
                # Set emissive factor to white (1,1,1) to show full texture colors
                material.emissiveFactor = [1.0, 1.0, 1.0]
                
                print(f"Applied emissive texture to material {i}")
            else:
                # No emissive
                material.emissiveFactor = [0.0, 0.0, 0.0]
        
        # Save the modified GLB file
        print(f"Saving modified GLB to {modified_glb_path}")
        gltf.save(modified_glb_path)
        
        # Return only the filename of the modified GLB (not the full path)
        output_filename = os.path.basename(modified_glb_path)
        
        # Return preview of emissive texture if available
        if emissive_preview is not None:
            return (output_filename, torch.from_numpy(emissive_preview))
        else:
            # Create a black preview image
            if hasattr(texture_image, 'cpu'):
                shape = texture_image[0].cpu().numpy().shape
            else:
                shape = texture_image[0].shape if len(texture_image.shape) == 4 else texture_image.shape
            
            black_mask = np.zeros(shape, dtype=np.float32)
            return (output_filename, torch.from_numpy(black_mask[None, ...]))


# Simplified presets for Unity-style
class QPresetGLBMaterialModifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "multiline": False}),
                "texture_image": ("IMAGE", {"forceInput": True}),
                "preset": (["Spaceship Metal", "Brushed Steel", "Chrome Hull", "Chrome Hull Lite", "Titanium", "Combat Metal", "Alien Tech"], {"default": "Spaceship Metal"}),
                "output_suffix": ("STRING", {"default": "_modified", "multiline": False}),
                "emissive_mode": (["Smart Auto", "Force On", "Force Off"], {"default": "Smart Auto"}),
                "absolute_min_brightness": ("INT", {"default": 128, "min": 50, "max": 200, "step": 1}),
                "render": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("modified_glb_path", "emissive_mask_preview")
    FUNCTION = "apply_material_preset"
    CATEGORY = "3d/q_material"

    def apply_material_preset(self, glb_path, texture_image, preset, output_suffix, emissive_mode, absolute_min_brightness, render):
        # Unity-style presets
        presets = {
            "Spaceship Metal": {
                "metallic": 0.85, "smoothness": 0.5, 
                "color": [0.7, 0.8, 0.9], "emissive_strength": 0.8,
                "brightness_threshold": 0.2, "percentage": 0.05
            },
            "Brushed Steel": {
                "metallic": 0.9, "smoothness": 0.3, 
                "color": [0.8, 0.8, 0.8], "emissive_strength": 0.6,
                "brightness_threshold": 0.25, "percentage": 0.03
            },
            "Chrome Hull": {
                "metallic": 1.0, "smoothness": 0.9, 
                "color": [0.95, 0.95, 0.95], "emissive_strength": 0.9,
                "brightness_threshold": 0.15, "percentage": 0.08
            },
            "Chrome Hull Lite": {
                "metallic": 1.0, "smoothness": 0.72, 
                "color": [0.95, 0.95, 0.95], "emissive_strength": 0.9,
                "brightness_threshold": 0.15, "percentage": 0.08
            },
            "Titanium": {
                "metallic": 0.8, "smoothness": 0.4, 
                "color": [0.6, 0.6, 0.7], "emissive_strength": 0.7,
                "brightness_threshold": 0.22, "percentage": 0.04
            },
            "Combat Metal": {
                "metallic": 0.7, "smoothness": 0.2, 
                "color": [0.5, 0.5, 0.6], "emissive_strength": 1.0,
                "brightness_threshold": 0.18, "percentage": 0.06
            },
            "Alien Tech": {
                "metallic": 0.6, "smoothness": 0.7, 
                "color": [0.4, 0.8, 0.6], "emissive_strength": 1.2,
                "brightness_threshold": 0.12, "percentage": 0.1
            }
        }
        
        settings = presets[preset]
        enable_emissive = emissive_mode != "Force Off"
        
        # Use the main node - pass all parameters
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
            enable_smart_emissive=enable_emissive,
            render=render
        )