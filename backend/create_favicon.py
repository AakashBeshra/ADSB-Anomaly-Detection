#!/usr/bin/env python3
"""
Create a simple ADS-B favicon using Python
"""

from PIL import Image, ImageDraw
import os

def create_favicon():
    print("🎨 Creating ADS-B favicon...")
    
    # Create images in multiple sizes
    sizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
    
    for size in sizes:
        # Create new image with transparent background
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        width, height = size
        
        # Colors
        bg_color = (26, 35, 126, 255)  # #1a237e
        aircraft_color = (0, 180, 216, 255)  # #00b4d8
        warning_color = (220, 38, 38, 255)  # #dc2626
        
        # Draw radar circle background (simplified for small sizes)
        if size[0] >= 32:
            # For larger sizes, draw radar
            radius = min(width, height) // 2 - 2
            center = (width // 2, height // 2)
            
            # Draw radar circle
            draw.ellipse([center[0]-radius, center[1]-radius, 
                         center[0]+radius, center[1]+radius], 
                         fill=bg_color, outline=aircraft_color)
            
            # Draw aircraft (triangle)
            aircraft_size = radius // 2
            draw.polygon([
                (center[0], center[1] - aircraft_size),  # Top
                (center[0] - aircraft_size//2, center[1] + aircraft_size//2),  # Bottom left
                (center[0] + aircraft_size//2, center[1] + aircraft_size//2)   # Bottom right
            ], fill=aircraft_color)
            
            # Draw warning dot for anomaly
            if size[0] >= 64:
                warning_size = radius // 4
                draw.ellipse([center[0]+radius//2, center[1]-radius//2,
                             center[0]+radius//2+warning_size, center[1]-radius//2+warning_size],
                             fill=warning_color)
        
        else:
            # For 16x16, ultra-simple design
            # Just a blue circle with red dot
            draw.ellipse([2, 2, width-2, height-2], fill=bg_color)
            draw.ellipse([width-5, 3, width-3, 5], fill=warning_color)
        
        # Save image
        filename = f"favicon_{size[0]}.png"
        img.save(filename)
        print(f"✅ Created {filename}")
    
    # Also create ICO file for browsers
    create_ico_file()
    
    print("\n📁 Files created:")
    print("  favicon_16.png - For browsers")
    print("  favicon_32.png - For high DPI")
    print("  favicon_64.png - For Apple devices")
    print("  favicon_128.png - For Chrome")
    print("  favicon_256.png - For high resolution")
    print("  favicon.ico - Combined ICO file")
    
    print("\n📝 Add to your HTML:")
    print("""<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon_16.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon_32.png">
<link rel="apple-touch-icon" sizes="180x180" href="/favicon_128.png">""")

def create_ico_file():
    """Create ICO file from PNGs"""
    try:
        from PIL import Image
        images = []
        
        for size in [16, 32, 48, 64]:
            try:
                img = Image.open(f"favicon_{size}.png")
                images.append(img)
            except:
                pass
        
        if images:
            images[0].save("favicon.ico", format='ICO', sizes=[(img.width, img.height) for img in images])
            print("✅ Created favicon.ico")
    except Exception as e:
        print(f"⚠️ Could not create ICO: {e}")

if __name__ == "__main__":
    create_favicon()