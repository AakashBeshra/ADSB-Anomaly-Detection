"""
Compatibility patch for Python 3.14
"""
import sys
import pkgutil
import warnings

# Monkey patch for pkgutil.ImpImporter if it doesn't exist
if not hasattr(pkgutil, 'ImpImporter'):
    print("⚠️  Python 3.14 detected - applying compatibility patches...")
    
    # Create a dummy ImpImporter class for compatibility
    class ImpImporter:
        def __init__(self, *args, **kwargs):
            pass
        
        def find_module(self, *args, **kwargs):
            return None
        
        def load_module(self, *args, **kwargs):
            return None
    
    # Add it to pkgutil
    pkgutil.ImpImporter = ImpImporter
    
    # Also patch setuptools if needed
    try:
        import setuptools
        import setuptools.version
        
        # Patch pkg_resources if it tries to use ImpImporter
        import pkg_resources
        
        # Replace the problematic registration
        original_register_finder = pkg_resources.register_finder
        
        def patched_register_finder(importer_type, loader):
            if importer_type.__name__ == 'ImpImporter':
                # Skip registration for ImpImporter
                warnings.warn("Skipping ImpImporter registration (Python 3.14 compatibility)")
                return
            return original_register_finder(importer_type, loader)
        
        pkg_resources.register_finder = patched_register_finder
        
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Compatibility warning: {e}")

# Optional: Add other Python 3.14 compatibility patches here
def check_python_version():
    """Check and warn about Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 14:
        print(f"✅ Running on Python 3.{version.minor} - compatibility mode active")
        return True
    return False

# Run version check on import
check_python_version()