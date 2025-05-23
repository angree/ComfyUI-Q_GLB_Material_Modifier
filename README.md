# ComfyUI-Q_GLB_Material_Modifier

A ComfyUI custom node for enhancing GLB 3D models generated with Hunyuan 3D 2.0. This module helps solve the common "plastic look" problem by adding realistic materials and smart emissive elements.

## Features

- Transforms the default "plastic" appearance of Hunyuan 3D 2.0 models into realistic materials
- Automatically detects bright areas in textures and converts them to emissive parts
- Preserves the original colors of the emissive areas
- Properly embeds emissive textures directly into the GLB file
- Easy to use with presets for common materials
- Works with any GLB model from Hunyuan 3D 2.0

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-Q_GLB_Material_Modifier.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```
or install manually:
```bash
pip install pygltflib>=1.16.1 opencv-python>=4.8.0 pillow>=10.0.0 numpy>=1.24.0
```

3. Restart ComfyUI

## Dependencies

This module requires the following Python libraries:
- **pygltflib**: For GLB/GLTF file manipulation
- **opencv-python**: For image processing operations
- **pillow**: For image loading and conversion
- **numpy**: For array operations

A `requirements.txt` file is included for easy installation.

## Usage with Hunyuan 3D 2.0

The extension provides two main nodes specifically designed for enhancing Hunyuan 3D 2.0 outputs:

### Q Manual GLB Material Modifier

This node provides full control over material parameters:

- **glb_path**: Path to your Hunyuan 3D 2.0 GLB file
- **texture_image**: Input texture from Hunyuan 3D 2.0
- **output_suffix**: Suffix to add to the output filename (default: "_modified")
- **metallic_factor**: How metallic the surface should be (0-1)
- **smoothness**: How smooth the surface should be (0-1)
- **base_color_r/g/b**: Base color tint
- **emissive_brightness_threshold**: Threshold for detecting bright areas
- **emissive_percentage**: Percentage of the brightest pixels to make emissive
- **absolute_min_brightness**: Minimum brightness to consider emissive
- **emissive_strength**: Brightness multiplier for emissive areas
- **enable_smart_emissive**: Enable/disable emissive effect
- **render**: Enable/disable processing (when disabled, skips GLB generation)

### Q Preset GLB Material Modifier

This node provides convenient presets specifically designed for Hunyuan 3D 2.0 models:

- **glb_path**: Path to your Hunyuan 3D 2.0 GLB file
- **texture_image**: Input texture from Hunyuan 3D 2.0
- **preset**: Choose from predefined material presets:
  - Spaceship Metal
  - Brushed Steel
  - Chrome Hull
  - Titanium
  - Combat Metal
  - Alien Tech
- **output_suffix**: Suffix to add to the output filename (default: "_modified")
- **emissive_mode**: Smart Auto, Force On, or Force Off
- **absolute_min_brightness**: Minimum brightness to consider emissive
- **render**: Enable/disable processing (when disabled, skips GLB generation)

## Hunyuan 3D 2.0 Workflows

In the `workflows` subfolder, you'll find example ComfyUI workflows specifically for Hunyuan 3D 2.0:

### Fixing the "Plastic Look" Problem

Hunyuan 3D 2.0 models (using DIT 2.0 fp16) typically have a default "plastic" appearance:
- The preset modifiers are specifically designed to transform this plastic look into realistic materials
- Chrome Hull preset works well for sci-fi and mechanical objects
- Brushed Steel creates a more industrial look
- Titanium is good for high-tech and aerospace models
- Adjust the metallic_factor and smoothness in the Manual modifier for fine control

### Batch Generation

Generate multiple variants of a Hunyuan 3D 2.0 model with different materials:
- Connect multiple Q Preset GLB Material Modifier nodes to the same texture
- Set different output_suffix values for each variant (e.g., "_chrome", "_steel")
- Use the render parameter to selectively enable only the variants you want to generate
- This workflow is efficient for testing different material looks without regenerating the 3D model

### Emission Map Generation

Create glowing elements in your Hunyuan 3D 2.0 models:
- Adjust emissive_brightness_threshold to control which areas will glow
- Use emissive_percentage to fine-tune the amount of emissive areas
- The emissive texture preserves the original colors from the Hunyuan texture
- For sci-fi models, try the "Alien Tech" preset with emissive_mode set to "Force On"

## License

[MIT License](LICENSE)
# ComfyUI-Q_GLB_Material_Modifier

A ComfyUI custom node for adding smart emissive materials to GLB 3D models. Perfect for adding glowing elements to your 3D models based on the brightest parts of textures generated with AI.

## Features

- Automatically detects bright areas in textures and converts them to emissive parts
- Preserves the original colors of the emissive areas
- Properly embeds emissive textures directly into the GLB file
- Easy to use with presets for common material types
- Works with any GLB model and any texture

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-Q_GLB_Material_Modifier.git
```

2. Install required dependencies:
```bash
pip install pygltflib pillow opencv-python
```

3. Restart ComfyUI

## Usage

The extension provides two main nodes:

### Q Manual GLB Material Modifier

This node provides full control over material parameters:

- **glb_path**: Path to your GLB file
- **texture_image**: Input texture (e.g. from AI generation)
- **output_suffix**: Suffix to add to the output filename (default: "_modified")
- **metallic_factor**: How metallic the surface should be (0-1)
- **smoothness**: How smooth the surface should be (0-1)
- **base_color_r/g/b**: Base color tint
- **emissive_brightness_threshold**: Threshold for detecting bright areas
- **emissive_percentage**: Percentage of the brightest pixels to make emissive
- **absolute_min_brightness**: Minimum brightness to consider emissive
- **render**: Enable/disable processing (when disabled, skips GLB generation)
- **emissive_strength**: Brightness multiplier for emissive areas
- **enable_smart_emissive**: Enable/disable emissive effect
- **render**: Enable/disable processing (when disabled, skips GLB generation)

### Q Preset GLB Material Modifier

This node provides convenient presets for common materials:

- **glb_path**: Path to your GLB file
- **texture_image**: Input texture (e.g. from AI generation)
- **preset**: Choose from predefined material presets:
  - Spaceship Metal
  - Brushed Steel
  - Chrome Hull
  - Titanium
  - Combat Metal
  - Alien Tech
- **output_suffix**: Suffix to add to the output filename (default: "_modified")
- **emissive_mode**: Smart Auto, Force On, or Force Off
- **absolute_min_brightness**: Minimum brightness to consider emissive

## Example Workflow

1. Generate a texture using your favorite image generation node
2. Connect the generated image to the `texture_image` input of the Q Preset GLB Material Modifier node
3. Set the path to your GLB file
4. Choose a preset that matches your desired look
5. Run the workflow to get a modified GLB with emissive materials

## License

[MIT License](LICENSE)