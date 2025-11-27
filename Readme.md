# LCC Tool for Blender

LCC Tool is a Blender add-on designed to import and render LCC (Lightweight Compressed Cloud) splat data. It allows users to load `.lcc` files, visualize them using Geometry Nodes, and set up high-quality 360° panoramic renders using the Cycles engine.

## Features

*   **LCC Import**: Reads `.lcc` files along with their associated `Index.bin` and `Data.bin` data.
*   **LOD Support**: Selectable Level of Detail (LOD) to balance between quality and performance.
*   **Geometry Nodes Rendering**: Automatically sets up a Geometry Nodes modifier to render splats as 3D ellipsoids.
*   **Custom Shader**: Generates a `GaussianSplatMat` material with Gaussian falloff for accurate splat visualization.
*   **360° Render Setup**: Optionally configures a panoramic camera and optimizes Cycles settings for transparency and heavy particle counts.
*   **User-Friendly Panel**: Accessible via the "LCC Tools" tab in the 3D View sidebar.

## Requirements

*   **Blender**: Version 3.3 or higher (Recommended for Geometry Nodes compatibility).
*   **Python Dependencies**: `numpy` (Standard with Blender).

## Installation

1.  Download `LCC_Render.py`.
2.  Open Blender.
3.  Go to **Edit > Preferences > Add-ons**.
4.  Click **Install...** and select `LCC_Render.py`.
5.  Enable the add-on by checking the box next to **Import-Export: LCC Tools**.
    *   *Alternatively, you can open the script in the Scripting workspace and run it directly.*

## Usage

1.  In the 3D Viewport, press `N` to open the sidebar.
2.  Click on the **LCC Tools** tab.
3.  Click the **Import LCC (.lcc)** button.
4.  Navigate to and select your `.lcc` file.
5.  Adjust the import settings in the file browser sidebar (bottom left) or popup:
    *   **LOD Level**: Choose the detail level (0 is highest, but heaviest).
    *   **Splat Scale Multiplier**: Adjust the size of splats to fill gaps (default: 1.5).
    *   **Min Thickness**: Set a minimum thickness to prevent splats from disappearing at glancing angles.
    *   **Setup 360 Render**: Check this to automatically create a panorama camera and optimize render settings.
6.  Click **Import LCC** to finish.

## Import Settings

*   **LOD Level**: Controls the density of the imported cloud.
    *   `0`: Full resolution (Heavy).
    *   `1-5`: Lower resolutions for faster viewport performance.
*   **Splat Scale Multiplier**: Multiplies the scale of every splat. Higher values create a solid surface look but may appear "blobby".
*   **Min Thickness**: Enforces a minimum scale on the smallest axis of the splat ellipsoid. Useful for preventing artifacts when looking at flat splats from the side.
*   **Setup 360 Render**:
    *   Switches render engine to **Cycles**.
    *   Creates a **PanoramaCam** at (0,0,0).
    *   Sets resolution to 4096x2048.
    *   Increases transparency bounces to handle many overlapping semi-transparent splats.

## Technical Details

The importer reads a custom binary format where splat data (Position, Color, Scale, Rotation) is distributed across `Index.bin` and `Data.bin`.
*   **Rotations** are decoded from a packed uint32 format into Quaternions and then Euler angles.
*   **Rendering** is handled by instancing Ico Spheres on points using Geometry Nodes. The visual appearance is controlled by a shader that calculates a Gaussian alpha falloff from the center of each instance.
